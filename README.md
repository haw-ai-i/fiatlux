# Fiatlux Benchmark Toolkit

[![build](https://github.com/haw-ai-i/fiatlux/actions/workflows/build.yml/badge.svg)](https://github.com/haw-ai-i/fiatlux/actions/workflows/build.yml)
[![style](https://github.com/haw-ai-i/fiatlux/actions/workflows/style.yml/badge.svg)](https://github.com/haw-ai-i/fiatlux/actions/workflows/style.yml)

![](../media/fiatlux_banner.png)

The **Fiatlux Benchmark** is an internal standalone benchmark for evaluating AI and robotics algorithms on energy infrastructure maintenance tasks. Specifically, the benchmark tests a robot's ability to replace light bulbs and climb ladders, facilitating research in complex manipulation, locomotion, and eventual sim-to-real transfer.

## Benchmark Hierarchy & Experimentation Plan

To systematically build up to the full end-to-end task, the benchmark is structured as a hierarchy of increasingly complex tasks:

### Level 1: Core Subtasks (Isolated Evaluation)
Before attempting the complete routine, algorithms can be developed and validated on isolated subtasks:
* **Ladder Climbing**: Locomotion and balance control for the bipedal G1 robot to climb industrial ladders.
* **Light Bulb Removal**: Grasping and unscrewing a faulty bulb from a socket.
* **Light Bulb Insertion**: Grasping and screwing a new bulb securely into a socket.
* **Verification & Inspection**: Visual and sensory checks to confirm bulb status and socket placement.

### Level 2: Combined Task (End-to-End Maintenance)
The ultimate benchmark task integrates all Level 1 capabilities into a single, contiguous trial:
* **Combined Light Bulb Replacement**: The robot must navigate the environment, locate a faulty lamp, perform visual inspection, climb the ladder to reach the socket, remove the old bulb, insert the new one, and verify successful replacement.

## Evaluation Metrics

Performance is measured quantitatively across the different levels of the benchmark using the following metrics:

### Level 1 Subtask Metrics
* **Ladder Climbing**:
  - *Climb Success*: Binary indicator of whether the robot reached the target height.
  - *Stability Index*: Average deviation of the center of mass (CoM) and peak contact forces on the ladder rungs (to evaluate fall risk).
  - *Climb Speed*: Vertical velocity ($\text{m/s}$) during the climb.
* **Light Bulb Removal & Insertion**:
  - *Success Rate*: Fraction of trials where the bulb is successfully removed (unscrewed/detached) or inserted (screwed/seated).
  - *Force/Torque Safety Margin*: Maximum contact forces and torques exerted during manipulation (to ensure no bulb/socket breakage).
  - *Pose Accuracy*: Position/orientation deviation from the target socket seating.
* **Verification & Inspection**:
  - *Classification Accuracy*: Precision and recall of detecting bulb types and socket states (empty vs. occupied).
  - *Inspection Time*: Time taken to complete the visual/sensory check.

### Level 2 Combined Task Metrics
* **End-to-End Task Success**: Binary indicator of whether the entire replacement routine is completed successfully.
* **Completion Time**: Total duration (seconds) taken to navigate, climb, remove, insert, and verify.
* **Locomotion and Manipulation Efficiency**: Integrated control effort (sum of joint torques squared) and trajectory length.
* **Collision Penalty**: Number of unintended collisions with the ladder, fixture, or environment.

---

## Benchmark Guide

This repository contains the toolkit to configure the simulation environments, run evaluation trials, and develop policies.

1. **📖 Understand the Benchmark**
   - Read the [Benchmark Overview](./docs/overview.md) to understand the goals.
   - Review the [Scoring Guide](./docs/scoring.md) to understand how policies are evaluated.

2. **🔧 Set Up Your Environment**
   - Follow the [Getting Started](./docs/getting_started.md) guide to set up your local workspace with Pixi.
   - Run the asset sync script to download required mesh assets.

3. **💻 Develop Your Policy**
   - Explore the [Scene Description](./docs/scene_description.md) to learn how to customize the environment.
   - Review [Fiatlux Interfaces](./docs/fiatlux_interfaces.md) to understand communication with sensors and actuators.
   - Consult [Fiatlux Controller](./docs/fiatlux_controller.md) to learn about controlling the robot.
   - Start with the [Policy Integration Guide](./docs/policy.md) to implement your solution.

4. **🧪 Test Your Solution**
   - Use the provided Isaac Lab simulation environment to test your policy.
   - Run `fiatlux_engine` with the `sample_config` in [`fiatlux_engine/config/`](./fiatlux_engine/config/) to test different scenarios.

5. **📅 Project Timeline & Roadmaps**
   - Review the unified [Project Timeline & Milestones](./docs/timeline_and_milestones.md) for the 12-week schedule.
   - Explore the individual technical plans:
     - [Phase 1: Infrastructure Setup](./docs/phase_1_plan.md)
     - [Phase 2: Simulation Policy Training](./docs/phase_2_plan.md)
     - [Phase 3: Sim-to-Real Gap Closure](./docs/phase_3_plan.md)

---

## Toolkit Architecture

The Fiatlux Benchmark toolkit is divided into two main components:

### 1. Evaluation Component
This component provides the evaluation and orchestration infrastructure:
- **`fiatlux_engine`** - Orchestrates trials, sends tasks, and computes scores.
- **`fiatlux_bringup`** - Launches the Isaac Lab simulation environment (robot, sensors, and scene assets).
- **`fiatlux_controller`** - Low-level robot control.

### 2. User Policy Component (Your Implementation)
This is what you develop and evaluate:
- **`fiatlux_model`** - A ROS 2 Lifecycle node that responds to the `/replace_bulb` action and outputs robot motion commands.
- We provide a convenient policy framework that handles the ROS 2 boilerplate. You simply implement a Python policy class that gets dynamically loaded at runtime. See the [Policy Integration Guide](./docs/policy.md) for details.

---

## Repository Structure

```
fiatlux/
├── fiatlux_assets/           # Scripts to download and manage 3D meshes (G1, bulbs, lamps, ladders)
├── fiatlux_bringup/          # Launch configurations for the Isaac Lab environment
├── fiatlux_controller/       # Robot joint and cartesian controller interfaces
├── fiatlux_description/      # URDF/SDF descriptions for the robot and fixtures
├── fiatlux_engine/           # Trial orchestration and scoring logic
├── fiatlux_policies/         # Baseline and example policy implementations
├── fiatlux_interfaces/       # ROS 2 message, service, and action definitions
├── fiatlux_model/            # Policy runner node framework
├── fiatlux_scoring/          # Evaluators for checking task success
├── fiatlux_utils/            # Training, teleoperation, and Isaac Lab wrappers
├── docker/                   # Docker setups for reproducible evaluation
└── docs/                     # Documentation templates
```

---

## Support and Resources

- **Issues**: Report any bugs or technical issues via [GitHub Issues](https://github.com/haw-ai-i/fiatlux/issues).
- **Access Errors**: If you encounter GCP bucket permissions errors when running the asset download scripts, reach out directly to **molybog@hawaii.edu**.

---

## License

This project is licensed under the Apache License 2.0. The `fiatlux_isaac` folder contains files licensed under BSD-3.
