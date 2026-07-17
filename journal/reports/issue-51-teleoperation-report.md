# Teleoperation harness — progress report (issue #51)

**Goal of the issue:** stand up a human-in-the-loop teleoperation tool so a person can drive the G1
by hand and **validate the scene physics** (can the bulb be grasped? does it seat? do contacts /
friction behave?) *before* investing in RL or imitation training.

**Status:** a keyboard/SpaceMouse manipulation teleop runs in the actual Insert benchmark scene, and
a separate script adds real GR00T-policy walking. The biggest finding is a UX one: **driving precise
manipulation from the keyboard is genuinely hard** — the hand rarely moves the way you expect.

---

## 1. What we built

- **`FIATLUX-Insert-Teleop-v0` + `scripts/insert_teleop.py`** — the Insert task with its action
  interface swapped from joint-position to **absolute differential-IK on the right arm** + a
  **binary open/close grip**. It runs *inside the real benchmark scene* (same assets, observations,
  rewards), so it tests the exact physics we ship. Keyboard is the default device; SpaceMouse /
  gamepad are supported (stock Isaac Lab devices). Fixed base (bolted pelvis) so the robot stands.
- **Grabbable bulb** — the Omniverse bulb shipped colliders but *no rigid body* and was USD-instanced,
  so it couldn't be picked up. De-instancing it at spawn and applying a runtime rigid body + mass
  made it a real dynamic manipuland; the socket stays kinematic as the fixture.
- **Real walking add-on** — `scripts/insert_with_walk_teleop.py`: a free-base G1 driven by NVIDIA
  GR00T's pretrained Balance/Walk ONNX policies (external `gr00t_wbc` dependency), so the robot
  **steps and balances** on arrow-key commands *while* the human drives the arm — walk up to the
  bench, then reach in. (Standalone demo, not the benchmark env.)

## 2. The hard part: keyboard teleoperation doesn't move the way you expect

This is the main lesson from actually driving it. Coarse "get the hand roughly over there" is fine;
**precise grasp/insert is clumsy**, for several compounding reasons:

- **Wrong frame.** The hand keys (`W/S A/D Q/E`) move the end-effector in the **robot's body frame**,
  not the camera/screen frame. So "forward" is forward *for the robot*, not "away from me on screen" —
  and once the robot or camera is at an angle, the controls feel rotated/backwards. You're constantly
  translating directions in your head.
- **One axis at a time.** A keyboard gives **discrete, single-axis** nudges. Real manipulation wants
  smooth, **simultaneous 6-DoF** (translate *and* rotate together to line the grip up). Tapping X,
  then Q, then C to compose a pose is slow and jerky — you can't "reach and twist to align" in one
  motion the way a hand naturally does.
- **The arm won't go where you point it.** The fixed-base arm has a **small, near-singular tabletop
  workspace**: the wrist can't pitch far down (top-down grasps freeze the IK), and near the reach
  limit the solver bifurcates into unreachable "dead zones." So you press a key and the hand simply
  **doesn't move there** — it sticks, drifts sideways, or snaps — which reads as "the robot isn't
  listening."
- **Runaway targets.** Because the IK target integrates your key presses, pushing past the arm's
  reach made the target **run away** past the hand; you'd have to "unwind" the overshoot before the
  hand responded again (felt stuck). We added a **leash** that clamps the target near the actual hand
  so reversing responds immediately — a real improvement, but the underlying bandwidth problem stays.
- **Grasping a small object is fiddly.** Positioning the hand, orienting it, and closing the grip at
  the right instant on a ~6 cm bulb is very error-prone from keys — it's easy to nudge the bulb away
  before the fingers close.

**Fixes we made to make it usable (not to make it good):** absolute IK so the hand holds against
gravity instead of sagging; the target leash; disabling the Inspire hand's self-collision (its finger
meshes interpenetrate at spawn and made the fingers spasm — the "random" jitter); disabling automatic
terminations so bumping the bulb doesn't reset the scene; a follow-camera + keyboard zoom so the hand
stays in view. These made it *drivable*, but the core conclusion holds:

> **Keyboard is a low-bandwidth, frame-mismatched input for 6-DoF manipulation.** It's adequate for
> coarse repositioning and physics *probing*, but poor for precise grasp/insert. The fix is to
> **use VR to move** — your hands drive the robot's hands directly (all axes at once), so it moves the
> way you expect — which is also the right path for collecting clean manipulation demos.

## 3. What this validated about the physics

Driving by hand did its job — it surfaced real, attributable scene issues that a trained policy would
have hit silently: the bulb needed a rigid body to be graspable; the hand's self-collision was
unstable; and the **fixed-base arm's reach is too limited** for comfortable tabletop manipulation
(which is exactly what motivated the walking add-on — let the robot reposition instead of over-reach).

## 4. Next

- **Use VR to move** the hands directly, so manipulation feels natural (the keyboard-difficulty fix).
- Instrumentation: on-screen contact-force / grasp-success readouts so a teleop pass is a recorded
  pass/fail, not just "looked right."
- LeRobot-format recording so a validated session doubles as imitation-training data.
