"""Demo recording for teleop sessions, built on the benchmark's own bag writer.

``TrajectoryRecorder`` (``fiatlux_task.recording``) writes robomimic-style HDF5 bags
(``data/demo_<i>`` episode groups) -- the same format ``record_run.py`` writes for policy
rollouts. Recorded per step: policy_obs, the teleop action, the complete commanded joint
vector (``joint_pos_target``, incl. the SONIC legs), measured joint states, object poses,
contact forces, reward, and termination flags. Two things stop the parent recorder from
working on teleop as-is, both fixed here WITHOUT touching the benchmark file:

* it is Insert-specific: ``record_step`` hard-reads ``bulb``/``socket``/``hand_contact`` and the
  bulb-socket error terms, which the Carry / LadderGallery scenes don't have. This subclass
  probes the scene once and records those fields only where they exist, so one recorder serves
  every teleop task;
* it segments episodes on the env's ``done`` flag -- which never fires in the teleop envs
  (their cfgs disable auto-terminations so a live session is never reset under the operator).
  ``mark_episode_end()`` closes the current episode at the operator's own boundary (the [R]
  reset, or quitting): in teleop, the human IS the termination condition.

Note on action spaces: the recorded ``actions`` are the TELEOP env's (arm-IK EE pose + binary
grip), not the benchmark env's whole-body action; ``joint_pos_target`` carries the joint-level
commands for consumers that need those instead.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import fiatlux_task.recording as _rec
import numpy as np
from fiatlux_task.recording import TrajectoryRecorder, term_flag
from fiatlux_task.viz import VideoRecorder, _draw_overlay


class StreamingVideoRecorder(VideoRecorder):
    """The benchmark's :class:`VideoRecorder` buffers every frame in RAM -- fine for its bounded
    clips, fatal for an open-ended teleop session (~2.7 MB/frame at 720p = GBs per minute).
    This variant streams each frame straight into the encoder: constant memory, any length."""

    def __init__(self, env, camera, out_path: str, fps: int = 50):
        super().__init__(env, camera, out_path, fps=fps)
        self._writer = None
        self._n = 0
        self._poster: np.ndarray | None = None

    def capture(self, pose: tuple | None = None, overlay: str | None = None) -> None:
        if pose is not None:
            self.set_pose(*pose)
        rgb = self._cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        if overlay:
            rgb = _draw_overlay(rgb, overlay)
        if self._writer is None:
            import imageio.v2 as imageio

            os.makedirs(os.path.dirname(self._out_path) or ".", exist_ok=True)
            self._writer = imageio.get_writer(self._out_path, fps=self._fps, codec="libx264", quality=8)
        self._writer.append_data(rgb)
        if self._n == 100 or self._poster is None:  # a settled early frame; frame 0 as fallback
            self._poster = rgb
        self._n += 1

    def __len__(self) -> int:
        return self._n

    def write(self) -> str:
        if self._writer is None:
            return ""
        self._writer.close()
        import imageio.v2 as imageio

        poster = self._out_path.rsplit(".", 1)[0] + "_poster.png"
        imageio.imwrite(poster, self._poster)
        return self._out_path


def _np_asarray(v) -> np.ndarray:
    return np.asarray(v).copy()


def _embed_score(out_dir) -> dict | None:
    """Score the just-written bag with the benchmark's own scorer and merge the result into
    meta.json (key ``"score"``), so a demo bag carries its benchmark score breakdown the same
    way a policy run does. Best-effort: scoring must never take a teleop session down."""
    try:
        import importlib.util
        import sys

        score_py = Path(__file__).resolve().parents[3] / "scripts" / "score.py"
        spec = importlib.util.spec_from_file_location("_fiatlux_score", score_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load score spec from {score_py}")
        mod = importlib.util.module_from_spec(spec)
        # Must be registered BEFORE exec: score.py's @dataclass resolves types through
        # sys.modules[cls.__module__], which is None for an unregistered manual load.
        sys.modules["_fiatlux_score"] = mod
        spec.loader.exec_module(mod)
        episodes, meta = mod.load_bag(out_dir)
        result = mod.score_bag(episodes, meta, mod.ScoreConfig())
        meta_path = Path(out_dir) / "meta.json"
        meta_disk = json.loads(meta_path.read_text())
        meta_disk["score"] = result
        meta_path.write_text(json.dumps(meta_disk, indent=2))
        return result
    except Exception as e:  # noqa: BLE001
        print(f"[teleop_recording] WARNING: could not embed score in meta.json ({e})", flush=True)
        return None


def _write_score_report(out_dir, info: dict, gate_seconds: float) -> None:
    """Explain the take's score in plain text, next to the bag.

    A bare 0.00 tells the operator nothing about WHY. This reconstructs the verdict from the
    recorded gate columns: which conjunct held, for how long, which one broke the sustained run
    and how often, and what the penalties did to the final number.
    """
    import glob

    try:
        import h5py
    except ImportError:
        return
    bags = glob.glob(os.path.join(out_dir, "run.h5")) + glob.glob(os.path.join(out_dir, "run.npz"))
    if not bags or not bags[0].endswith(".h5"):
        return
    sc = info.get("score") or {}
    cfg = sc.get("score_config") or {}
    lines = []
    try:
        with h5py.File(bags[0], "r") as f:
            demos = sorted(f["data"].keys())
            need = max(1, round(gate_seconds / 0.02)) if gate_seconds else 0
            lines.append(
                f"score: {sc.get('mean_score')}   success_rate: {sc.get('success_rate')}   episodes: {len(demos)}"
            )
            lines.append("")
            for k in demos:
                ep = f["data"][k]
                n = len(np.array(ep["success_term"]))
                succ = bool(np.array(ep["success_term"]).any())
                lines.append(f"--- {k}: {n} steps ({n * 0.02:.1f}s) -> {'SUCCESS' if succ else 'no success'}")
                cols = sorted(c for c in ep if c.startswith("gate_"))
                if not cols:
                    lines.append("    (no gate columns recorded)")
                    continue
                vals = {c: np.array(ep[c]).astype(bool) for c in cols}
                allc = np.ones(n, dtype=bool)
                for c in cols:
                    allc &= vals[c]
                for c in cols:
                    a = vals[c]
                    idx = np.flatnonzero(np.diff(np.r_[0, a.astype(np.int8), 0]))
                    runs = (idx[1::2] - idx[::2]) if len(idx) else np.array([0])
                    lines.append(
                        f"    {c.replace('gate_', ''):26s} held {100.0 * a.sum() / n:5.1f}%"
                        f"   longest {runs.max() * 0.02:5.2f}s"
                    )
                idx = np.flatnonzero(np.diff(np.r_[0, allc.astype(np.int8), 0]))
                runs = (idx[1::2] - idx[::2]) if len(idx) else np.array([0])
                lines.append(
                    f"    ALL TOGETHER               longest {runs.max() * 0.02:5.2f}s   (needs {gate_seconds:.2f}s)"
                )
                if not succ and need:
                    breaks = np.flatnonzero(allc[:-1] & ~allc[1:]) + 1
                    who = {c: sum(1 for b in breaks if not vals[c][b]) for c in cols}
                    worst = [
                        f"{c.replace('gate_', '')} ({k2}x)" for c, k2 in sorted(who.items(), key=lambda x: -x[1]) if k2
                    ]
                    if worst:
                        lines.append(f"    WHY NOT: the run was broken {len(breaks)} time(s) by " + ", ".join(worst))
                    elif runs.max() == 0:
                        never = [c.replace("gate_", "") for c in cols if not vals[c].any()]
                        lines.append(
                            "    WHY NOT: never satisfied at once; never true at all: "
                            + (", ".join(never) if never else "(all held at some point)")
                        )
            lines.append("")
            pen = []
            if sc.get("broken_rate"):
                pen.append(
                    f"BROKEN  -{cfg.get('broken_penalty', 1.0)} "
                    f"(peak contact {sc.get('peak_contact_force', 0):.1f} N > "
                    f"{cfg.get('fragility_threshold', 50)} N)"
                )
            if sc.get("dropped_rate"):
                pen.append(
                    f"DROPPED -{cfg.get('dropped_penalty', 1.0)} (payload below {cfg.get('drop_min_height', 0.4)} m)"
                )
            lines.append("penalties: " + ("; ".join(pen) if pen else "none"))
            lines.append(
                f"arithmetic: 1.0 per successful episode, minus penalties, floored at "
                f"{cfg.get('min_score', 0.0)}, averaged over {len(demos)} episode(s)"
                f"  ->  {sc.get('mean_score')}"
            )
    except Exception as e:  # noqa: BLE001
        lines.append(f"(report incomplete: {e!r})")
    Path(out_dir, "score_report.txt").write_text("\n".join(lines) + "\n")


class ImageCapture:
    """Optional per-camera frame capture for image-based training (ACT and friends).

    Discovers every RGB camera sensor the scene already carries (``wrist_camera``,
    ``ego_camera``, ...; the follow-cam ``video_cam`` is excluded -- that one feeds the
    human-review MP4) and writes JPEGs to ``<session>/images/<camera>/f<index>.jpg`` at a
    decimated stride. ``index`` is the RECORDED-step counter, so frames align 1:1 with rows of
    the flat bag stream -- split them into episodes with the bag's ``done`` column, exactly like
    the numeric data."""

    def __init__(self, env, out_dir: str, stride: int = 5, exclude: tuple = ("video_cam",)):
        self._cams = {}
        for name, sensor in getattr(env.scene, "sensors", {}).items():
            if name in exclude:
                continue
            out = getattr(getattr(sensor, "data", None), "output", None)
            try:
                if out is not None and "rgb" in out:
                    self._cams[name] = sensor
            except Exception:  # noqa: BLE001 - a sensor without dict-like output is not a camera
                continue
        self._dir = out_dir
        self._stride = max(1, int(stride))
        self._written = 0
        for name in self._cams:
            os.makedirs(os.path.join(out_dir, name), exist_ok=True)

    @property
    def cameras(self) -> list[str]:
        return sorted(self._cams)

    def __len__(self) -> int:
        return self._written

    def maybe_capture(self, step_index: int) -> None:
        if not self._cams or step_index % self._stride:
            return
        import imageio.v2 as imageio

        for name, cam in self._cams.items():
            rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            imageio.imwrite(os.path.join(self._dir, name, f"f{step_index:06d}.jpg"), rgb, quality=90)
        self._written += 1


class TeleopTrajectoryRecorder(TrajectoryRecorder):
    """Task-agnostic, operator-segmented variant of the benchmark bag recorder."""

    def __init__(self, env, *, policy_spec: str, seed: int, checkpoint: str | None = None):
        super().__init__(env, policy_spec=policy_spec, seed=seed, checkpoint=checkpoint)

        def _has(name: str) -> bool:
            try:
                env.scene[name]
                return True
            except Exception:  # noqa: BLE001 - scene raises KeyError/ValueError depending on version
                return False

        # Bulbs are named by placement since #76 Step 1. The parent resolved which one this task
        # manipulates; reuse that rather than guessing a second time.
        self._has_insert = _has(self._bulb_entity) and _has("socket")
        self._has_contact = "hand_contact" in getattr(env.scene, "sensors", {})
        # Both hands (#89). Teleop is where this matters most: an operator uses whichever hand is
        # convenient, and the 2026-08-21 session drove the old bulb left-handed, which the bags
        # recorded as 0.0 N of contact throughout.
        self._has_left_contact = "left_hand_contact" in getattr(env.scene, "sensors", {})

        # The parent's meta hardcodes action_joint_order from the benchmark's STATIC constants
        # (Inspire hand names) -- wrong whenever the operator picked the other hand (--hand
        # dex3|inspire swaps the actions/robot at cfg time). Rebuild it from the LIVE action
        # manager so the meta always matches the robot that actually ran, and keep the per-term
        # breakdown too. Best-effort: meta polish must never take a session down.
        try:
            am = env.action_manager
            terms: dict[str, list[str]] = {}
            for name in am.active_terms:
                term = am.get_term(name) if hasattr(am, "get_term") else am._terms[name]
                terms[name] = list(getattr(term, "_joint_names", []) or [])
            if terms:
                self._meta["action_terms"] = terms
                self._meta["action_joint_order"] = [j for names in terms.values() for j in names]
        except Exception as e:  # noqa: BLE001
            print(
                f"[teleop_recording] WARNING: could not resolve live action joints ({e}); "
                "meta keeps the parent's static action_joint_order",
                flush=True,
            )

        # Columns this layer adds on top of the parent recorder. Kept at the END of
        # __init__, away from the hand/bulb flags above: main edits those same lines, and
        # a block butted straight up against them turns every such edit into a conflict.
        self._gate_hold = 0
        self._gate_fired = False
        self._gate_need = (
            max(1, round(self._gate_seconds / float(getattr(env, "step_dt", 0.02))))
            if self._gate_seconds
            else (1 if self._gate_conjuncts else 0)
        )
        if self._gate_conjuncts:
            print(
                "[teleop_recording] recording gate conjuncts: "
                + ", ".join(getattr(f, "__name__", "?") for f, _ in self._gate_conjuncts),
                flush=True,
            )
        else:
            print("[teleop_recording] WARNING: no success gate found -- takes cannot score", flush=True)

    def _resolve_gate(self, env) -> tuple[list, float]:
        """The gate the driver stashed, falling back to the live ``success`` term.

        The driver clears the termination so a success cannot reset the scene mid-take, which
        leaves the parent's lookup with nothing to unwrap.
        """
        cfg = getattr(env, "cfg", None)
        params = getattr(cfg, "teleop_success_spec", None)
        if params is None:
            return super()._resolve_gate(env)
        try:
            from fiatlux_task.tasks.manager_based.fiatlux_task.mdp.gates import conjuncts_of

            fn = getattr(cfg, "teleop_success_fn", None)
            return list(conjuncts_of(fn, params)), float(params.get("seconds") or 0.0)
        except Exception:  # noqa: BLE001
            return [], 0.0

    def record_step(self, obs, actions, reward, terminated, truncated, extras=None) -> None:
        env = self.env
        robot = env.scene["robot"]
        done = terminated | truncated
        body_state = robot.data.body_state_w[:, self._ee_id, :7]  # (N, 7) pos+quat

        step = {
            "actions": actions,
            # The COMPLETE commanded joint vector -- includes what bypasses the action manager
            # (sonic_teleop writes SONIC leg targets straight to the articulation), so walking is
            # in the record, not just the 16-dim arm/grip action.
            "joint_pos_target": robot.data.joint_pos_target,
            "joint_pos": robot.data.joint_pos,
            "joint_vel": robot.data.joint_vel,
            "joint_acc": robot.data.joint_acc,
            "eef_pose": body_state,
            **(
                {"eef_pose_left": robot.data.body_state_w[:, self._left_ee_id, :7]}
                if getattr(self, "_left_ee_id", None) is not None
                else {}
            ),
            "policy_obs": obs["policy"] if isinstance(obs, dict) else obs,
            "reward": reward,
            "step_in_episode": env.episode_length_buf.clone(),
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
            "success_term": term_flag(env, "success", self.n, self.device),
            "dropped_term": term_flag(env, "bulb_dropped", self.n, self.device),
            "timeout_term": term_flag(env, "time_out", self.n, self.device),
        }
        if self._has_insert:
            bulb, socket = env.scene[self._bulb_entity], env.scene["socket"]
            step.update(
                {
                    "bulb_pos": bulb.data.root_pos_w,
                    "bulb_quat": bulb.data.root_quat_w,
                    "bulb_lin_vel": bulb.data.root_lin_vel_w,
                    "socket_pos": socket.data.root_pos_w,
                    "socket_quat": socket.data.root_quat_w,
                    "pos_error": _rec._rewards._bulb_socket_pos_error(env, self._bulb_entity),
                    "ori_error": _rec._rewards._bulb_socket_ori_error(env, self._bulb_entity),
                }
            )
        _gate = self.gate_fields()
        step.update(_gate)
        # The take's own success verdict. The driver clears the termination, so the manager's flag
        # would read False forever; this sustains the parent's gate columns over the same window
        # mdp.gates.sustained uses, then LATCHES -- scripts/score.py reads success_term on the
        # episode's LAST step, and without the latch a demo that achieved the task and kept going
        # would score 0.
        if self._gate_need:
            _all_true = all(bool(v.reshape(-1)[0]) for v in _gate.values())
            self._gate_hold = self._gate_hold + 1 if _all_true else 0
            if self._gate_hold >= self._gate_need:
                self._gate_fired = True
            step["success_term"] = np.full((self.n,), self._gate_fired, dtype=bool)
        step.update(self.object_state_fields())
        step.update(self.contact_fields())
        if self._has_contact:
            step["contact_force"] = _rec._obs.object_contact_forces(env.scene.sensors["hand_contact"])
        if self._has_left_contact:
            step["contact_force_left"] = _rec._obs.object_contact_forces(env.scene.sensors["left_hand_contact"])
        # Robot root + grip force, shared with the RL recorder (issues #107, #106).
        # This class builds its own step dict rather than extending the parent's, so the parent's
        # fields have to be merged in explicitly -- they do not arrive by inheritance.
        step.update(self.world_state_fields())
        if extras:
            # Driver-supplied operator/policy signals (loco_cmd, SONIC leg action, ...). Tensors or
            # numpy accepted; each must already carry the (N, ...) leading env axis.
            step.update(extras)

        for key, value in step.items():
            arr = _rec._np(value) if hasattr(value, "detach") else _np_asarray(value)
            # Force a real copy. On a CPU-device env (the XR path falls back to CPU physics)
            # tensor.to("cpu") is a no-op and .numpy() ALIASES the source buffer, so a field backed
            # by a persistent in-place-updated tensor records the flush-time value on EVERY row --
            # which is why the ladder's velocity columns read identically zero while the live gate
            # saw it moving.
            self._buf.setdefault(key, []).append(np.array(arr, copy=True))

    def write(self, out_dir, *, fmt: str = "hdf5") -> dict:
        info = super().write(out_dir, fmt=fmt)
        score = _embed_score(out_dir)
        if score is not None:
            info["score"] = score
        _write_score_report(out_dir, info, getattr(self, "_gate_seconds", 0.0))
        return info

    def reset_gate(self) -> None:
        """Clear the sustain counter and latch. Called when a take ends, so the next take is judged
        on its own, not on a success carried over from the previous one."""
        self._gate_hold = 0
        self._gate_fired = False

    def reset_buffers(self) -> None:
        """Drop everything buffered.

        For per-take bag mode: the driver writes each closed take to its own ``epNN/`` folder,
        then clears the buffer so the next take starts a fresh bag. Without this, ``write()``
        re-emits every prior take into every later folder.
        """
        self._buf = {}

    def mark_episode_end(self) -> bool:
        """Flag the LAST buffered step as the end of an episode (operator reset / quit).

        The parent's ``episodes()`` splits the stream on ``done``; teleop envs never set it, so
        without this every session would serialize to zero episodes. Returns False when nothing
        has been buffered yet (e.g. reset pressed before the first step).
        """
        if not self._buf.get("done"):
            return False
        for key in ("truncated", "done"):
            last = self._buf[key][-1].copy()
            last[:] = True
            self._buf[key][-1] = last
        return True
