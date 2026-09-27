# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Psi-0 (SONIC) zero-shot baseline adapter.

``psi0`` -- USC PSI Lab's Psi-0 VLA (Qwen3-VL-2B backbone + flow-matching action expert),
the released SONIC fine-tune ``psi0/sonic-checkpoints/multi-task.psi-dream.2609092156``
(post-trained on 50 h of UnifoLM G1 data retargeted to SONIC tokens, fine-tuned on 5 h of
Psi-Dream G1 + Dex3 teleop), served by ``scripts/psi0/serve.sh`` over HTTP ``POST /act``.
Every 0.5 s it sees one head-camera RGB frame, 45-D proprioception, and a language
instruction, and returns 15 rows of a 30 Hz, 80-D action chunk: a 64-D GEAR-SONIC motion
token, 14 Dex3 finger targets, and 2 neck targets. The token drives the in-process
:class:`~fiatlux_task.groot.SonicDecoder` (the released SONIC v1.0 decoder, zero-order hold
from 30 Hz onto the env's 50 Hz ticks -- what the deploy stack's ZMQ token stream does); the
finger targets go straight to the Dex3 joints; the neck targets are dropped (this G1 has no
neck, its head camera is fixed).

The wire contract is read from the checkpoint's own ``run_config.json`` / ``argv.txt``, not
guessed -- see ``docs/psi0_baseline.md`` for where each piece comes from and for every
judgment call made wiring it up.

Standard-mode note: like ``groot``, everything consumed is sensor-realizable -- the raw
head-camera frame, joint encoders -- read from scene handles because the model needs raw
values, not the normalized ``policy`` group.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch

from .groot import _BODY_JOINT_NAMES, _TOKEN_DIM, DEFAULT_INSTRUCTION, STAND_TOKEN, SonicDecoder, _StartupBlend

DEFAULT_PSI0_ENDPOINT = "localhost:8014"

# The checkpoint's repack config: ``image_keys``, ``dataset_name``, ``pad_state_dim``,
# ``action_dim``, and the model resize (``data.transform.model.resize.size``, H x W).
PSI0_IMAGE_KEY = "observation.images.head"
PSI0_DATASET_NAME = "g1sonic0810"
PSI0_STATE_DIM = 45
PSI0_ACTION_DIM = 80
PSI0_IMAGE_HW = (270, 480)
_PSI0_HZ = 30  # the Psi-Dream / UnifoLM LeRobot packs are 30 fps
_ENV_HZ = 50  # SonicDecoder's control rate (it asserts env.step_dt == 0.02)

# Psi-0's 45-D state for this checkpoint is the "legacy g1_sonic_lerobot_0810 order"
# (Psi0 ``scripts/data/backfill_joint_names.py``: ``--joint-order leg waist arm hand neck``),
# the 29 body joints in URDF order, then both Dex3 hands thumb -> middle -> index (Psi0
# ``scripts/data/merge_posttrain_sonic.py``: "Both hands are thumb -> MIDDLE -> INDEX"), then
# the neck. The 80-D action is ``action[16:80] ++ action[:14] ++ action[14:16]`` of the pack
# (``argv.txt``'s ``--data.transform.repack.action-keys``): token, the same 14 hand joints in
# the same order, neck.
_PSI0_HAND_ORDER = [
    "hand_thumb_0_joint",
    "hand_thumb_1_joint",
    "hand_thumb_2_joint",
    "hand_middle_0_joint",
    "hand_middle_1_joint",
    "hand_index_0_joint",
    "hand_index_1_joint",
]
PSI0_HAND_JOINT_NAMES = [f"left_{j}" for j in _PSI0_HAND_ORDER] + [f"right_{j}" for j in _PSI0_HAND_ORDER]
PSI0_STATE_JOINT_NAMES = list(_BODY_JOINT_NAMES) + PSI0_HAND_JOINT_NAMES  # + 2 neck slots, zero here
_NECK_DIM = 2
assert len(PSI0_STATE_JOINT_NAMES) + _NECK_DIM == PSI0_STATE_DIM
assert _TOKEN_DIM + len(PSI0_HAND_JOINT_NAMES) + _NECK_DIM == PSI0_ACTION_DIM

# The flow head predicts a continuous token; Psi-0's own robot client snaps it onto SONIC's FSQ
# grid before publishing it to the controller (Psi0 ``real/SONIC/run_psi0_rtc_sonic_dex1.py``
# -> ``src/psi/deploy/mock_psi0_client_rtc.py:fsq_quantize``), the grid the decoder was trained on.
_FSQ_MIN, _FSQ_MAX, _FSQ_STEP = -0.625, 0.625, 0.0625

# One plain sentence per subtask, written from each subtask's own module docstring, plus the
# full task's canonical sentence. Psi-0 was trained with per-task instructions (its pooled CLIP
# cache holds ~100 of them, e.g. "pick up the paper ball and turn left and throw it into the
# trash can"), so a whole-task paragraph for a single leg would be off-distribution for no
# benefit. Override with --instruction.
TASK_INSTRUCTIONS = {
    "FIATLUX-Replace-v0": DEFAULT_INSTRUCTION,
    "FIATLUX-S01-MoveLadder-v0": (
        "walk to the ladder, pick it up, carry it to the light fixture and stand it up under the fixture"
    ),
    "FIATLUX-S02-ClimbLadder-v0": "climb up the ladder to the top steps",
    "FIATLUX-S03-RemoveOldBulb-v0": "grasp the light bulb in the ceiling fixture and pull it out",
    "FIATLUX-S04-DescendWithBulb-v0": "climb down the ladder while holding the light bulb",
    "FIATLUX-S05-CarryBulbToDisposal-v0": "carry the light bulb to the yellow crate",
    "FIATLUX-S06-DisposeBulb-v0": "put the light bulb into the yellow crate and let go of it",
    "FIATLUX-S07-ApproachNewBulb-v0": "walk to the light bulb on the table",
    "FIATLUX-S08-GrabNewBulb-v0": "pick up the light bulb from the table",
    "FIATLUX-S09-CarryBulbToLadder-v0": "carry the light bulb to the ladder",
    "FIATLUX-S10-ClimbWithBulb-v0": "climb up the ladder while holding the light bulb",
    "FIATLUX-S11-ScrewInBulb-v0": "insert the light bulb into the ceiling fixture",
    "FIATLUX-S12-ClimbDown-v0": "climb down the ladder",
}


def instruction_for_task(task: str | None) -> str:
    """The default Psi-0 instruction for a task id (``-Training`` variants share their base's)."""
    if task is None:
        return DEFAULT_INSTRUCTION
    return TASK_INSTRUCTIONS.get(task.replace("-Training-v0", "-v0"), DEFAULT_INSTRUCTION)


def configure_env_cfg(env_cfg) -> None:
    """Render the head camera in the 16:9 geometry Psi-0 was trained on, before the env is built.

    The benchmark's ego camera is 256x256 at the RealSense D435's 69.4 deg horizontal FOV (sized
    for GR00T's preprocessing). Psi-0 resizes every frame to 270x480 without preserving aspect,
    so a square frame would reach it stretched 1.78x horizontally. Rendering 480x270 at the same
    horizontal aperture (vertical aperture follows the aspect ratio) is the D435's own 16:9
    colour mode -- 69.4 x 42.5 deg -- so the sensor stays a real one and the model sees no stretch.
    """
    cam = getattr(env_cfg.scene, "ego_camera", None)
    if cam is None:
        raise ValueError("psi0 needs the head-mounted ego_camera; this task's scene does not attach one")
    cam.height, cam.width = PSI0_IMAGE_HW
    if getattr(cam.spawn, "vertical_aperture", None) is not None:
        cam.spawn.vertical_aperture = cam.spawn.horizontal_aperture * PSI0_IMAGE_HW[0] / PSI0_IMAGE_HW[1]


def chunk_row(steps_into_chunk: int) -> int:
    """30 Hz chunk row in effect ``steps_into_chunk`` 50 Hz env ticks after the chunk arrived.

    Zero-order hold: row ``r`` covers ``[r/30, (r+1)/30)`` s, so tick ``n`` (at ``n/50`` s) reads
    row ``floor(n * 30 / 50)``.
    """
    return (steps_into_chunk * _PSI0_HZ) // _ENV_HZ


def ticks_per_rows(rows: int) -> int:
    """50 Hz env ticks until a ``rows``-row 30 Hz segment is exhausted (the next query tick)."""
    return -(-rows * _ENV_HZ // _PSI0_HZ)


def fsq_quantize(token: np.ndarray) -> np.ndarray:
    """Snap a motion token onto SONIC's FSQ grid (``mock_psi0_client_rtc.fsq_quantize``)."""
    q = np.round(np.clip(token, _FSQ_MIN, _FSQ_MAX) / _FSQ_STEP) * _FSQ_STEP
    return np.clip(q, _FSQ_MIN, _FSQ_MAX).astype(np.float32)


def _np_blob(array: np.ndarray) -> dict:
    """Psi0 ``psi.deploy.helpers.numpy_serialize``: raw bytes, numpy dtype descr, shape."""
    array = np.ascontiguousarray(array)
    return {
        "__numpy__": base64.b64encode(array.tobytes()).decode(),
        "dtype": np.lib.format.dtype_to_descr(array.dtype),
        "shape": list(array.shape),
    }


def _np_unblob(blob: dict) -> np.ndarray:
    """Inverse of :func:`_np_blob` (``psi.deploy.helpers.numpy_deserialize``)."""
    array = np.frombuffer(base64.b64decode(blob["__numpy__"]), np.lib.format.descr_to_dtype(blob["dtype"]))
    return array.reshape(blob["shape"]) if blob["shape"] else array[0]


class _Psi0Client:
    """Minimal client for Psi0's HTTP policy server (``serve_psi0_simple.Server``: POST /act)."""

    def __init__(self, endpoint: str, timeout_s: float = 120.0):
        self._base = f"http://{endpoint}"
        self._timeout = timeout_s

    def info(self) -> dict:
        with urllib.request.urlopen(f"{self._base}/info", timeout=self._timeout) as resp:
            return json.load(resp)

    def act(self, image: np.ndarray, state: np.ndarray, instruction: str, history: dict) -> np.ndarray:
        payload = {
            "image": {PSI0_IMAGE_KEY: _np_blob(image)},
            "instruction": instruction,
            "history": history,
            "state": {"states": _np_blob(state)},
            "condition": {},
            "gt_action": [],
            "dataset_name": PSI0_DATASET_NAME,
            "timestamp": str(time.time()),
        }
        request = urllib.request.Request(
            f"{self._base}/act",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as resp:
            body = json.load(resp)
        # The server's catch-all returns HTTP 200 with a JSON *string* ('{"status": "<error>"}')
        # when inference raises, so a 200 alone does not mean an action came back.
        if not isinstance(body, dict) or "action" not in body:
            raise RuntimeError(f"Psi-0 server error (see the server log for the traceback): {body!r}")
        return _np_unblob(body["action"]).astype(np.float32)


class Psi0Policy:
    """Psi-0 SONIC checkpoint + in-process GEAR-SONIC decoder, on the Dex3 G1.

    Per query (every 15 Psi-0 rows = 25 env ticks, from the tick after the one-tick startup
    hold): sends the current head-camera frame, the 45-D joint state, and the instruction;
    receives a (15, 80) chunk. Each 50 Hz tick then feeds the held row's token to
    :class:`SonicDecoder` (body) and its 14 finger targets to the Dex3 joints. The first query of
    each episode sends ``history["reset"]`` so the server's test-time RTC does not splice onto
    the previous episode's chunk.

    No standing-latent settle before the first query, although Psi-0's real-robot procedure
    settles SONIC in its standing reference first: half the subtasks START with the bulb in hand,
    and a settle under SONIC's standing latent swings the arms to SONIC's rest pose and opens the
    fingers to their defaults, which drops the bulb before Psi-0 ever acts (S06 then ends at
    step ~48, exactly like the ``zero`` policy). The fingers instead hold their spawn pose until
    the first chunk arrives, and the body holds its spawn pose for the one startup tick.

    Exposes the optional ``info`` telemetry dict (``policy/`` namespace): server round-trip,
    query count, and the current token's magnitude.
    """

    def __init__(self, env, endpoint: str | None = None, instruction: str | None = None):
        assert env.num_envs == 1, "the Psi-0 adapter is single-env (the server keeps one RTC session)"
        self._env = env
        self.robot = env.scene["robot"]
        device = env.device
        self._instruction = instruction or DEFAULT_INSTRUCTION
        self._camera = env.scene["ego_camera"]

        missing = [n for n in PSI0_HAND_JOINT_NAMES if n not in self.robot.joint_names]
        if missing:
            # Every frame Psi-0 was trained on is a Dex3 G1; there is nothing to map 7-DoF
            # finger targets onto on the Inspire hand, and dead hands would read as the
            # policy failing the task.
            raise ValueError(f"psi0 drives the Dex3 hand (run with --robot dex3); robot lacks {missing}")

        self._decoder = SonicDecoder(env)
        self._blend = _StartupBlend(
            env, self._decoder.robot, self._decoder.joint_ids, self._decoder.sonic_default, self._decoder.action_scale
        )
        # The decoder still runs during the startup hold (it keeps its state histories fed), but
        # its output is discarded there, so the token it gets does not matter.
        self._stand = torch.from_numpy(STAND_TOKEN).to(device).expand(env.num_envs, -1)

        state_ids, _ = self.robot.find_joints(PSI0_STATE_JOINT_NAMES, preserve_order=True)
        self._state_ids = torch.tensor(state_ids, dtype=torch.long, device=device)
        hand_ids, _ = self.robot.find_joints(PSI0_HAND_JOINT_NAMES, preserve_order=True)
        self._hand_ids = torch.tensor(hand_ids, dtype=torch.long, device=device)

        endpoint = endpoint or DEFAULT_PSI0_ENDPOINT
        self._client = _Psi0Client(endpoint)
        try:
            server = self._client.info()
        except Exception as exc:
            raise RuntimeError(
                f"no Psi-0 policy server reachable at {endpoint}; start it with scripts/psi0/serve.sh"
            ) from exc
        action = server["action"]
        assert action["action_dim"] == PSI0_ACTION_DIM, f"server serves a {action['action_dim']}-D action head"
        assert PSI0_IMAGE_KEY in server["expected_keys"]["image"], f"server expects images {server['expected_keys']}"
        self._rows = int(action["action_exec_horizon"])
        self._query_ticks = ticks_per_rows(self._rows)
        self._session = f"fiatlux-{os.getpid()}"

        hw = tuple(self._camera.data.output["rgb"].shape[1:3])
        if hw != PSI0_IMAGE_HW:
            print(
                f"[Psi-0] WARNING: ego camera renders {hw}, Psi-0 trains on {PSI0_IMAGE_HW} and resizes "
                "without keeping aspect -- build the env through psi0.configure_env_cfg (record_run.py does).",
                file=sys.stderr,
                flush=True,
            )

        self._chunk: np.ndarray | None = None
        self._hand_hold = torch.zeros(len(PSI0_HAND_JOINT_NAMES), device=device)
        self._ticks_into_chunk = 0
        self._reset_session = True
        self._queries = 0
        self.info: dict[str, float] = {}

    def _state(self) -> np.ndarray:
        q = self.robot.data.joint_pos[0, self._state_ids].cpu().numpy().astype(np.float32)
        return np.concatenate([q, np.zeros(_NECK_DIM, dtype=np.float32)])

    def _frame(self) -> np.ndarray:
        frame = self._camera.data.output["rgb"][0, ..., :3]
        if frame.dtype != torch.uint8:
            frame = frame.clamp(0, 255).to(torch.uint8)
        return frame.cpu().numpy()

    def _query(self) -> None:
        history = {"session_id": self._session}
        if self._reset_session:
            history["reset"] = True
        t0 = time.perf_counter()
        chunk = self._client.act(self._frame(), self._state(), self._instruction, history)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        assert chunk.shape == (self._rows, PSI0_ACTION_DIM) and np.isfinite(chunk).all(), (
            f"server returned a {chunk.shape} chunk (finite={np.isfinite(chunk).all()}); "
            f"expected ({self._rows}, {PSI0_ACTION_DIM})"
        )
        self._chunk = chunk
        self._ticks_into_chunk = 0
        self._reset_session = False
        self._queries += 1
        self.info = {"server_latency_ms": latency_ms, "queries": float(self._queries)}

    def __call__(self, obs) -> torch.Tensor:
        env = self._env
        if bool((env.episode_length_buf == 0).any()):
            self._chunk = None
            self._ticks_into_chunk = 0
            self._reset_session = True
            self._hand_hold = self.robot.data.joint_pos[0, self._hand_ids].clone()

        blend_mask = self._blend.mask(env)
        if bool(blend_mask.any()):
            tokens = self._stand
            hand_targets = self._hand_hold
        else:
            if self._chunk is None or self._ticks_into_chunk >= self._query_ticks:
                self._query()
            row = self._chunk[chunk_row(self._ticks_into_chunk)]
            self._ticks_into_chunk += 1
            tokens = torch.from_numpy(fsq_quantize(row[None, :_TOKEN_DIM])).to(env.device)
            hand_targets = torch.from_numpy(row[_TOKEN_DIM : _TOKEN_DIM + len(PSI0_HAND_JOINT_NAMES)]).to(env.device)
            self.info["token_abs_mean"] = float(np.abs(row[:_TOKEN_DIM]).mean())

        action = self._decoder.step(env, tokens, hold_mask=blend_mask)
        data = self.robot.data
        limits = data.soft_joint_pos_limits[0, self._hand_ids]
        hand_targets = hand_targets.clamp(limits[:, 0], limits[:, 1])
        hand_offsets = hand_targets - data.default_joint_pos[0, self._hand_ids]
        action[:, self._hand_ids] = hand_offsets / self._decoder.action_scale
        return self._blend.override(env, action, blend_mask)


def make_psi0_policy(env, endpoint: str | None = None, instruction: str | None = None, task: str | None = None):
    """The zero-shot Psi-0 SONIC baseline policy (instruction defaults to the task's sentence)."""
    return Psi0Policy(env, endpoint=endpoint, instruction=instruction or instruction_for_task(task))
