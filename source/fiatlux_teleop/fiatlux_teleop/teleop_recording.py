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

import numpy as np
from fiatlux_task import recording as _rec
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
        if self._n == 100 or self._poster is None:   # a settled early frame; frame 0 as fallback
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
            print(f"[teleop_recording] WARNING: could not resolve live action joints ({e}); "
                  "meta keeps the parent's static action_joint_order", flush=True)

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
            step.update({
                "bulb_pos": bulb.data.root_pos_w,
                "bulb_quat": bulb.data.root_quat_w,
                "bulb_lin_vel": bulb.data.root_lin_vel_w,
                "socket_pos": socket.data.root_pos_w,
                "socket_quat": socket.data.root_quat_w,
                "pos_error": _rec._rewards._bulb_socket_pos_error(env, self._bulb_entity),
                "ori_error": _rec._rewards._bulb_socket_ori_error(env, self._bulb_entity),
            })
        if self._has_contact:
            step["contact_force"] = _rec._obs.object_contact_forces(env.scene.sensors["hand_contact"])
        if self._has_left_contact:
            step["contact_force_left"] = _rec._obs.object_contact_forces(
                env.scene.sensors["left_hand_contact"]
            )
        if extras:
            # Driver-supplied operator/policy signals (loco_cmd, SONIC leg action, ...). Tensors or
            # numpy accepted; each must already carry the (N, ...) leading env axis.
            step.update(extras)

        for key, value in step.items():
            self._buf.setdefault(key, []).append(
                _rec._np(value) if hasattr(value, "detach") else _np_asarray(value))

    def write(self, out_dir, *, fmt: str = "hdf5") -> dict:
        info = super().write(out_dir, fmt=fmt)
        score = _embed_score(out_dir)
        if score is not None:
            info["score"] = score
        return info

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
