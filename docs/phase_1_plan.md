# Fiatlux Project Phase 1 Plan: Humanoid Energy Benchmark Setup

This document outlines the formalized plan for Phase 1 of the Fiatlux Benchmark, pivoting from the previous manufacturing arm task to a humanoid-based energy infrastructure maintenance benchmark (specifically climbing ladders, replacing light bulbs, and performing visual inspection).

> [!IMPORTANT]
> **Scope of Phase 1**: The objective of Phase 1 is strictly to prepare and verify the simulation, control, and orchestration infrastructure. This phase does *not* include training or deploying complex learned policies. Phase 1 is complete when the entire pipeline is operational and verified by running a simple, heuristic, or dummy policy through the evaluation orchestrator.

---

## 1. System Architecture & Control Strategy

### Locomotion & Dexterity
Rather than writing locomotion and whole-body controllers from scratch, the project will leverage existing controllers:
* **Whole-Body Control (WBC)**: Integrate **Groot Whole-Body Control** to manage the locomotion, balancing, and reach dexterity of the Unitree G1 humanoid robot.
* **Controller Interface**: Adapt the existing `fiatlux_controller` to bind with the Groot WBC SDK or library endpoints.

### Policy Action / Control Space
To ensure flexibility and support various learning paradigms, the policy's action space will support multiple target modes:
1. **Operational Space Cartesian Targets (Primary focus)**: Commanding target end-effector poses for arms/legs.
2. **Joint Position Targets**: Commanding target angles directly for all joints.
3. **Joint Torque Targets**: Commanding torques directly for fine-grained control and dynamics.

We will support all three modes and experiment to compare their convergence and stability during policy training.

### Policy Observation Space & "Cheat Code" Mode
To facilitate progressive training (sim-to-real and reinforcement learning bootstrap), the observation space will support two modes:
1. **Ground-Truth "Cheat Code" Mode**: Fully expose the true states of the environment (exact ladder rung positions, lamp socket coordinates, bulb pose relative to the gripper, etc.). This mirrors the ground-truth TF cheat-code regime from the AIC benchmark.
2. **Sensor/Visual-Only Mode**: Expose visual feeds (RGB-D cameras from head/wrists), proprioception, and IMU data, requiring policies to perform state estimation or learn visual end-to-end mappings.

---

## 2. Benchmark Task Hierarchy

### Level 1: Qualification Subtasks
* **Ladder Climbing**: G1 starts at the base of the ladder, climbs to a target height, and maintains stability on the rungs.
* **Light Bulb Removal**: G1 starts at the top of the ladder/platform, reaches, grasps, and unscrews/detaches a bulb from a fixture.
* **Light Bulb Insertion**: G1 starts with the bulb in the gripper, aligns it with the socket, and screws it in securely.
* **Verification & Inspection**: G1 uses visual sensors to classify the bulb type and check if the socket is empty or occupied.

### Level 2: Combined Task
* **End-to-End Replacement**: G1 must search for/navigate to the ladder, climb to the correct height, remove a burnt-out bulb, insert a new one, verify the task completion visually, and return safely.

---

## 3. Implementation Plan by Component

### Component 1: Simulation & Asset Composition
* **Pull G1 & Inspire Hands Assets**: Locate, pull, and aggregate the raw USD/URDF files for the Unitree G1 humanoid robot and the Inspire dexterous hands from internal systems/repositories.
* Assemble the G1 robot (with Inspire hands attached), ladder, lamp, and bulb USD files into a composite world asset: `fiatlux_g1_scene.usd`.
* Define specific reference coordinate frames and contact sensors (hands, feet, rungs, bulb screw thread).
* Update `fiatlux_task_env_cfg.py` to target the G1 humanoid and specify observation, action, and termination limits (e.g. timeout, fall detection).
* Add domain randomization in `mdp/events.py` to vary ladder incline, lamp socket placement, and initial bulb position during reset.

### Component 2: Robot Control & SDK Integration
* Update `fiatlux_controller` to load and interface with the Groot WBC.
* Map incoming policy actions (Cartesian, joint position, or torque targets) to the underlying control loops.

### Component 3: Orchestration and Scoring (C++ Engine)
* Update `fiatlux_engine` C++ state machine to parse and spawn ladders and fixtures instead of task boards.
* Implement fall termination (detect when G1's base coordinates drop below a safety threshold or register off-limit collisions).
* Update `fiatlux_scoring` to calculate:
  - **Stability Index**: Center of Mass (CoM) tracking and contact force distribution.
  - **Force/Torque Safety Margin**: Peak wrenches experienced during screwing/unscrewing.
  - **Visual Verification Accuracy**: Classification metrics against ground truth.

---

## 4. Verification & Validation

### Phase 1 Milestone Verification
1. **Asset Load**: Confirm that the composite `fiatlux_g1_scene.usd` loads cleanly in Isaac Sim without physics or coordinate issues.
2. **Teleoperation & WBC Test**: Teleoperate the G1 robot in the scene using a keyboard or gamepad. Verify the Groot WBC keeps the robot balanced and supports climbing steps.
3. **Orchestrator Walkthrough**: Execute `fiatlux_engine` with a dummy model node. Confirm it spawns the assets, starts the timer, and correctly transitions states to completion/timeout.
