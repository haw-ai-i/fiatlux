# Roadmap

The functional benchmark today is the **insertion** subtask (`FIATLUX-Insert-v0`).
Everything below is planned, not implemented — listed so the extension seams are
intentional.

## 1. Climbing subtask — `FIATLUX-Climb-v0`

G1 climbs a ladder to reach the fixture height. The scene side of this exists: the
ladder task family (`FIATLUX-{Base,Carry,Climb,Descend,Remove,Install}-v0`) is
registered as non-RL scene scaffolds sharing `g1_ladder_env_cfg.py`, with the
BEHAVIOR-1K `shfvtl` ladder (`fiatlux_task.assets.LADDER_USD`). What remains is the
task logic:

- Upgrade the scaffold to `ManagerBasedRLEnvCfg`; add a whole-body / locomotion
  action space (the G1 base is already free).
- Add a fall-detection termination (base height / orientation thresholds).
- Reward: progressive height + hand/foot–rung contact + CoM-sway penalty.

## 2. Combined task — `FIATLUX-Replace-v0`

End-to-end: approach → climb → insert → verify. Chains the two subtasks in one
episode with a staged curriculum (manipulation → climbing → full chain).

## 3. Learned-policy support

- Imitation pre-training (ACT / Diffusion) from teleop or scripted "cheat-code"
  demos, using the `privileged` observation group.
- The current PPO config (`agents/rsl_rl_ppo_cfg.py`) covers RL fine-tuning.
- If demo recording is re-added, keep its dataset tooling (LeRobot/HDF5) optional
  and out of the core install.

## 4. Sim-to-real (physical G1)

Added as a **separate optional deployment adapter**, never the old ROS/Zenoh
harness:

- **Action bridge:** policy joint-position targets → Unitree SDK joint commands.
- **Observation bridge:** real proprioception + wrist camera + F/T → the `policy`
  observation vector (the `privileged` group is sim-only).
- **Perception:** estimate socket/bulb pose (ArUco or segmentation) to replace the
  ground-truth poses the scripted baseline uses.
- **Safety:** E-stop + contact-force limits (the env already penalizes contact
  force, so a hardware threshold maps cleanly).

The two design constraints that keep this cheap — hardware-realizable actions and a
sensor-realizable default observation group — are already baked into
`g1_bulb_env_cfg.py`.
