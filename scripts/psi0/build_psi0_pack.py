# ruff: noqa: E501  -- runs in the Psi0 venv; the column tables read best one line per group
"""Write a Psi-0 SONIC LeRobot v2.1 pack from encoded teleop takes (stage B of the fine-tune pack).

Runs in the Psi0 repo's venv (``.venv-psi``: pandas / pyarrow / datasets), after
``encode_sonic_tokens.py`` (fiatlux venv) has written one ``<take>.npz`` per take at 50 Hz.

Schema mirrors Psi0's released ``USC-PSI-Lab/psi-data:sonic/unifolm_sonic_lerobot_*`` packs:
``observation.state`` (43: legs, waist, arms, Dex3 thumb->middle->index; the neck is not stored,
the SONIC repack zero-pads it to 45), ``action`` (36: Dex3 targets 14, arm targets 14, 8 unused
torso/base slots left at zero), ``action.body_token`` (64, the v1 1/16-grid token), one 30 fps
video, ``meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl, stats.json,
stats_psi0.json, modality.json}``. One deliberate difference: the video key is
``observation.images.head``, the key the SONIC checkpoints and ``fiatlux_task.psi0`` use.

Resampling 50 Hz -> 30 Hz: frame ``j`` (t = j/30 s) takes everything -- image, state, token,
hand targets -- from recorded step ``round(j * 50 / 30)``, so each row stays internally
consistent (tokens are FSQ codes; interpolating them would not produce valid codes).

Images: the teleop twin renders the ego camera 512x512 at the D435's 69.4 deg horizontal FOV;
the centre 512x288 crop is the D435's 16:9 geometry (69.4 x 42.5 deg), which is what the zero-shot
benchmark renders natively at 480x270, so the crop is scaled to 480x270 and stored at the model's
input size.

``stats_psi0.json`` keeps the fine-tuned checkpoint's own normalization bounds (read from its
``run_config.json``), widened only where the demos leave them: a 10-take pack's own min/max would
re-map every normalized dimension the action head was trained on.

    .venv-psi/bin/python build_psi0_pack.py --encoded ~/psi0_ft/encoded/success \
        --takes ~/psi0_ft/raw/success --out ~/psi0_ft/packs/s06_dispose_teleop \
        --ckpt-run-dir ~/tools/psi0_checkpoints/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156 \
        --task "put the light bulb into the yellow crate and let go of it"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Sequence, Value

FPS_SRC, FPS = 50, 30
H, W = 270, 480
STATE_DIM, ACTION_DIM, TOKEN_DIM, HAND_DIM = 43, 36, 64, 14

BODY = [
    *(f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *(f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")),
]  # fmt: skip
HAND = [f"{s}_hand_{j}_joint" for s in ("left", "right") for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]  # fmt: skip
ACTION_NAMES = HAND + BODY[15:] + ["torso_roll", "torso_pitch", "torso_yaw", "torso_height", "base_vx", "base_vy", "base_vyaw", "base_target_yaw"]  # fmt: skip
assert len(BODY) + len(HAND) == STATE_DIM and len(ACTION_NAMES) == ACTION_DIM


def source_steps(n_src: int) -> np.ndarray:
    n = int(np.floor((n_src - 1) * FPS / FPS_SRC)) + 1
    return np.minimum(np.round(np.arange(n) * FPS_SRC / FPS).astype(int), n_src - 1)


def read_frames(mp4: Path) -> np.ndarray:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(mp4)],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")  # fmt: skip
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def write_video(frames: np.ndarray, out: Path) -> None:
    """Centre 16:9 crop, scale to 480x270, 30 fps h264/yuv420p (what LeRobot's pyav reader expects)."""
    n, h, w, _ = frames.shape
    ch = int(round(w * H / W))
    top = (h - ch) // 2
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
        "-vf", f"crop={w}:{ch}:0:{top},scale={W}:{H}:flags=area", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-g", "2", "-crf", "18", str(out),
    ]  # fmt: skip
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def stats_of(a: np.ndarray) -> dict:
    a = np.asarray(a, np.float64).reshape(len(a), -1)
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
        "q01": np.quantile(a, 0.01, axis=0).tolist(), "q99": np.quantile(a, 0.99, axis=0).tolist(), "count": [len(a)],
    }  # fmt: skip


def checkpoint_bounds(run_dir: Path) -> dict[str, np.ndarray]:
    """The fine-tuned checkpoint's own bounds, split back into this pack's columns.

    Its model-order action is ``token(64) ++ hand(14) ++ neck(2)``; its state is the 45-D layout
    (43 stored here + 2 neck slots the repack pads).
    """
    field = json.loads((run_dir / "run_config.json").read_text())["data"]["transform"]["field"]
    amin, amax = np.array(field["action_min"]), np.array(field["action_max"])
    return {
        "token_min": amin[:64], "token_max": amax[:64], "hand_min": amin[64:78], "hand_max": amax[64:78],
        "state_min": np.array(field["state_min"]), "state_max": np.array(field["state_max"]),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoded", type=Path, required=True)
    ap.add_argument("--takes", type=Path, required=True, help="raw take folders (for ego.mp4)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt-run-dir", type=Path, required=True)
    ap.add_argument("--task", required=True)
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "meta").mkdir(parents=True)
    features = Features(
        {
            "observation.state": Sequence(Value("float32")),
            "action": Sequence(Value("float32")),
            "action.body_token": Sequence(Value("float32")),
            "timestamp": Value("float32"),
            "frame_index": Value("int64"),
            "episode_index": Value("int64"),
            "index": Value("int64"),
            "task_index": Value("int64"),
            "task_description": Value("string"),
            "next.done": Value("bool"),
        }
    )
    all_state, all_action, all_token, episodes, ep_stats, provenance = [], [], [], [], [], []
    cursor = 0
    for ep, npz in enumerate(sorted(args.encoded.glob("*.npz"))):
        d = np.load(npz)
        frames = read_frames(args.takes / npz.stem / "ego.mp4")
        n_src = len(d["token"])
        assert len(frames) == n_src, f"{npz.stem}: {len(frames)} video frames vs {n_src} recorded steps"
        src = source_steps(n_src)
        state = d["state"][src, :STATE_DIM]
        action = np.zeros((len(src), ACTION_DIM), np.float32)
        action[:, :HAND_DIM] = d["hand_target"][src]
        action[:, HAND_DIM:28] = d["state"][src, 15:29]  # executed arm pose; unused by the SONIC recipe
        token = d["token"][src]
        n = len(src)
        rows = [
            {
                "observation.state": state[i].tolist(),
                "action": action[i].tolist(),
                "action.body_token": token[i].tolist(),
                "timestamp": i / FPS,
                "frame_index": i,
                "episode_index": ep,
                "index": cursor + i,
                "task_index": 0,
                "task_description": args.task,
                "next.done": i == n - 1,
            }
            for i in range(n)
        ]
        pq = args.out / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        pq.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(rows, features=features).to_parquet(str(pq))
        write_video(
            frames[src], args.out / "videos" / "chunk-000" / "observation.images.head" / f"episode_{ep:06d}.mp4"
        )
        episodes.append(
            {"episode_index": ep, "tasks": [0], "length": n, "task_description": args.task, "source_take": npz.stem}
        )
        numeric = {
            "observation.state": state, "action": action, "action.body_token": token,
            "timestamp": np.arange(n) / FPS, "frame_index": np.arange(n), "episode_index": np.full(n, ep),
            "index": cursor + np.arange(n), "task_index": np.zeros(n),
        }  # fmt: skip
        ep_stats.append({"episode_index": ep, "stats": {k: stats_of(v) for k, v in numeric.items()}})
        all_state.append(state)
        all_action.append(action)
        all_token.append(token)
        provenance.append(f"{npz.stem}: {n_src} steps @50Hz -> {n} frames @30Hz")
        cursor += n

    state, action, token = (np.concatenate(x) for x in (all_state, all_action, all_token))
    stats = {
        "observation.state": stats_of(state), "action": stats_of(action), "action.body_token": stats_of(token),
        "timestamp": stats_of(np.concatenate([np.arange(e["length"]) / FPS for e in episodes])),
    }  # fmt: skip
    bounds = checkpoint_bounds(args.ckpt_run_dir)
    psi0 = json.loads(json.dumps(stats))
    report = {}

    def widen(key, lo, hi, data):
        new_lo, new_hi = np.minimum(lo, data.min(0)), np.maximum(hi, data.max(0))
        report[key] = {
            "dims_widened": int(((new_lo < lo) | (new_hi > hi)).sum()),
            "frac_values_outside_ckpt_bounds": float(((data < lo) | (data > hi)).mean()),
        }
        return new_lo, new_hi

    lo, hi = widen("action.body_token", bounds["token_min"], bounds["token_max"], token)
    psi0["action.body_token"]["min"], psi0["action.body_token"]["max"] = lo.tolist(), hi.tolist()
    lo, hi = widen("action[:14]", bounds["hand_min"], bounds["hand_max"], action[:, :HAND_DIM])
    psi0["action"]["min"][:HAND_DIM], psi0["action"]["max"][:HAND_DIM] = lo.tolist(), hi.tolist()
    # 45-D state bounds: the stored 43 (widened) plus the checkpoint's own neck bounds, so the
    # zero-padded neck normalizes exactly as it did for the checkpoint (a 43-D entry would be padded
    # with 0/0 bounds and read as a constant instead).
    lo, hi = widen("observation.state", bounds["state_min"][:STATE_DIM], bounds["state_max"][:STATE_DIM], state)
    psi0["observation.state"]["min"] = lo.tolist() + bounds["state_min"][STATE_DIM:].tolist()
    psi0["observation.state"]["max"] = hi.tolist() + bounds["state_max"][STATE_DIM:].tolist()

    meta = args.out / "meta"
    video_info = {"video.fps": float(FPS), "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}  # fmt: skip
    info = {
        "codebase_version": "v2.1", "robot_type": "g1", "total_episodes": len(episodes), "total_frames": cursor,
        "total_tasks": 1, "total_videos": len(episodes), "total_chunks": 1, "chunks_size": 1000, "fps": FPS,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "joint_order": "unifolm_sonic_lerobot order; neck (state 43:45, action 78:80) padded by the SONIC repack",
        "features": {
            "observation.images.head": {"dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channel"], "video_info": video_info},
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": BODY + HAND},
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": ACTION_NAMES},
            "action.body_token": {"dtype": "float32", "shape": [TOKEN_DIM], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "task_description": {"dtype": "string", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
        },
    }  # fmt: skip
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": args.task}) + "\n")
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ep_stats))
    (meta / "stats.json").write_text(json.dumps(stats, indent=2))
    (meta / "stats_psi0.json").write_text(json.dumps(psi0, indent=2))
    modality = {
        "state": {"joint_positions": {"original_key": "observation.state", "start": 0, "end": STATE_DIM, "absolute": True}},
        "action": {
            "hand_joints": {"original_key": "action", "start": 0, "end": 14, "absolute": True},
            "arm_joints": {"original_key": "action", "start": 14, "end": 28, "absolute": True},
            "body_token": {"original_key": "action.body_token", "start": 0, "end": 64, "absolute": True},
        },
        "video": {"head": {"original_key": "observation.images.head"}},
        "annotation": {"human_instruction": {"original_key": "task_description"}},
    }  # fmt: skip
    (meta / "modality.json").write_text(json.dumps(modality, indent=2))
    (meta / "bounds_report.json").write_text(json.dumps(report, indent=2))
    print("\n".join(provenance))
    print(f"{len(episodes)} episodes, {cursor} frames @ {FPS} fps -> {args.out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
