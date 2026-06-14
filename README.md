# Fiatlux Benchmark Toolkit

[![build](https://github.com/intrinsic-dev/fiatlux/actions/workflows/build.yml/badge.svg)](https://github.com/intrinsic-dev/fiatlux/actions/workflows/build.yml)
[![style](https://github.com/intrinsic-dev/fiatlux/actions/workflows/style.yml/badge.svg)](https://github.com/intrinsic-dev/fiatlux/actions/workflows/style.yml)

![](../media/fiatlux_banner.png)

The **Fiatlux Benchmark** is an open competition for developers and roboticists aimed at solving some of the hardest, high-impact problems in robotics and manufacturing.

This repository contains the official toolkit to help participants start developing their solutions. For registration details, official rules, and FAQs, please visit the [Fiatlux Benchmark event page](https://www.intrinsic.ai/events/ai-for-industry-challenge).

---

## Toolkit Guide

Welcome to the FIATLUX toolkit documentation. This guide walks you through the complete workflow for participating in the challenge — from understanding the requirements to submitting your solution.

Follow the sections below to navigate through each phase of the process.

1. **📖 Understand the Challenge**
   - Read the [Challenge Overview](./docs/overview.md) to understand the goals.
   - Review the [Qualification Phase](./docs/phases.md#qualification-phase-train-your-model) to understand what you'll be building.
   - Review the [Scoring Guide](./docs/scoring.md) to understand how you'll be scored.

2. **🔧 Set Up Your Environment**
   - Follow the [Getting Started](./docs/getting_started.md) guide to set up and validate your development environment.
   - Run the evaluation container and set up your local workspace with Pixi.

3. **💻 Develop Your Policy**
   - Explore the [Scene Description](./docs/scene_description.md) to learn how to customize and explore the environment.
   - Review [FIATLUX Interfaces](./docs/fiatlux_interfaces.md) to understand available interfaces to communicate with sensors and actuators.
   - Consult [FIATLUX Controller](./docs/fiatlux_controller.md) to learn about controlling the robot.
   - Consult the [Challenge Rules](./docs/challenge_rules.md) to ensure compliance.
   - Start with the [Policy Integration Guide](./docs/policy.md) to implement your solution.
   - See [Participant Utilities](./docs/participant_utilities.md) for a list of helpful tools.

4. **🧪 Test Your Solution**
   - Use the provided simulation environment to test your policy.
   - Run `fiatlux_engine` with the `sample_config` in [`fiatlux_engine/config/`](./fiatlux_engine/config/) to test different scenarios. For more information on running the `fiatlux_engine` with different configs, see the [fiatlux_engine README file](./fiatlux_engine/README.md).
   - Create your own test scenarios by following the configuration example in [`fiatlux_engine/config/`](./fiatlux_engine/config/) to run with `fiatlux_engine`.
   - Refer to [Troubleshooting](./docs/troubleshooting.md) if you encounter issues.

5. **📦 Submit Your Entry**
   - Package your solution following the [Submission Guidelines](./docs/submission.md).
   - Test your container locally before submitting following [these instructions](./docs/submission.md#verify-locally).
   - Submit through the official portal following [these instructions](./docs/submission.md#2-upload-your-image-to-our-registry).

---

## Toolkit Architecture

![FIATLUX Competition Components](../media/fiatlux_competition_components.png)

The Fiatlux Benchmark toolkit is divided into **two main components**:

### 1. Evaluation Component (Provided - Run by Organizers)

This component provides the complete evaluation infrastructure:
- **`fiatlux_engine`** - Orchestrates trials and computes scores.
- **`fiatlux_bringup`** - Launches simulation environment (Gazebo, robot, sensors).
- **`fiatlux_controller`** - Low-level robot control with force management.
- **`fiatlux_adapter`** - Sensor fusion and data synchronization.

**What you receive:** Standard ROS sensor topics providing camera images, joint states, force/torque measurements, and TF frames.

### 2. Participant Model Component (Your Implementation - What You Submit)

This is what you develop and submit:
- **A ROS 2 node** that follows the behavioral requirements defined in [Challenge Rules](./docs/challenge_rules.md).
- **Your custom logic** - Code to process sensor data and command the robot to replace bulbs.

**What you provide:** A container with a ROS 2 Lifecycle node named `fiatlux_model` that responds to the `/replace_bulb` action and outputs robot motion commands via standard ROS topics/services.

**Convenient Entry Point:** We provide an `fiatlux_model` framework that handles all the ROS 2 boilerplate and lifecycle management. You simply implement a Python policy class that gets dynamically loaded at runtime. See the [Policy Integration Guide](./docs/policy.md) for details.

### Development and Submission Workflow

> [!IMPORTANT]
> **ROS 2 Distribution:** The official evaluation of all submissions will be conducted using **ROS 2 Kilted Kaiju**. If you choose to develop or test your policy using a different ROS 2 distribution (e.g., Humble or Jazzy), it is entirely your responsibility to ensure compatibility and support. Please note that **inter-distro communication is not guaranteed and not officially supported**.

**Development Options:**
- Develop inside a container (recommended - matches evaluation environment).
- OR develop in native Ubuntu 24.04 environment (requires all dependencies).

**Submission Requirements:**
- Package your solution using the provided `fiatlux_model` Dockerfile.
- Submit your container - it must respond to standard ROS inputs and command the robot to replace bulbs.
- Your container interfaces with the evaluation component via ROS topics.

---
## Repository Structure

```
fiatlux/
├── fiatlux_adapter/          # Adapter for interfacing between model and controller
├── fiatlux_assets/           # 3D models and simulation assets
├── fiatlux_bringup/          # Launch files for starting the challenge environment
├── fiatlux_controller/       # Robot controller implementation
├── fiatlux_description/      # Robot and environment URDF/SDF descriptions
├── fiatlux_engine/           # Trial orchestration and validation engine
├── fiatlux_example_policies/ # Example policy implementations
├── fiatlux_gazebo/           # Gazebo-specific plugins and configurations
├── fiatlux_interfaces/       # ROS 2 message, service, and action definitions
├── fiatlux_model/            # Template for participant policy implementation
├── fiatlux_scoring/          # Scoring system implementation
├── fiatlux_utils/            # Utility packages and tools
├── docker/               # Docker container definitions
└── docs/                 # Comprehensive documentation
```

---

## Key Packages for Participants

### `fiatlux_model` - Convenient Policy Framework (Recommended)
This package provides a ready-to-use ROS 2 Lifecycle node that dynamically loads and executes your Python policy implementation. It handles all ROS 2 boilerplate, lifecycle management, and challenge rule compliance, allowing you to focus on implementing your policy logic.
- **Location**: `fiatlux_model/`.
- **Documentation**: [Policy Integration Guide](./docs/policy.md).
- **Tutorial**: [Creating a New Policy Node](./docs/policy.md#tutorial-creating-a-new-policy-node).

> **Note:** While we recommend using this framework, you may implement your own ROS 2 node from scratch as long as it adheres to the [Challenge Rules](./docs/challenge_rules.md).

### `fiatlux_interfaces` - Communication Protocols
Defines all ROS 2 messages, services, and actions used in the challenge.
- **Location**: `fiatlux_interfaces/`.
- **Documentation**: [FIATLUX Interfaces](./docs/fiatlux_interfaces.md).

### `fiatlux_example_policies` - Reference Implementations
Example policies demonstrating different approaches and techniques.
- **Location**: `fiatlux_example_policies/`.
- **README**: [fiatlux_example_policies/README.md](./fiatlux_example_policies/README.md).

### `fiatlux_bringup` - Launch the Environment
Launch files to start the simulation, robot, and scoring systems.
- **Location**: `fiatlux_bringup/`.
- **README**: [fiatlux_bringup/README.md](./fiatlux_bringup/README.md).

### `fiatlux_engine` - Trial Orchestrator
Manages trial execution, validates participant models, and collects scoring data.
- **Location**: `fiatlux_engine/`.
- **README**: [fiatlux_engine/README.md](./fiatlux_engine/README.md).

---

## Additional Documentation

### Challenge Information

* **[Challenge Overview](./docs/overview.md):** High-level summary of the competition goals and structure.
* **[Competition Phases](./docs/phases.md):** Details on Qualification, Phase 1, and Phase 2.
* **[Qualification Phase](./docs/qualification_phase.md):** Detailed technical overview of the qualification phase trials and scoring.
* **[Challenge Rules](./docs/challenge_rules.md):** Required behavior for participant models.
* **[Scoring](./docs/scoring.md):** Metrics and methods used to evaluate performance.
* **[Scoring Test Examples](./docs/scoring_tests.md):** Reproducible examples exercising each scoring tier with exact commands.

### Technical Documentation

* **[Getting Started](./docs/getting_started.md):** How to set up your local development environment.
* **[Policy Integration](./docs/policy.md):** Guide to implementing your policy in the `fiatlux_model` framework.
* **[FIATLUX Interfaces](./docs/fiatlux_interfaces.md):** ROS 2 topics, services, and actions available to your policy.
* **[FIATLUX Controller](./docs/fiatlux_controller.md):** Understanding the robot controller and motion commands.
* **[Scene Description](./docs/scene_description.md):** Technical details of the simulation environment.
* **[Task Board Description](./docs/task_board_description.md):** Physical layout and specifications of the task board.
* **[Troubleshooting](./docs/troubleshooting.md):** Common issues and debugging strategies.

### Reference Materials

* **[Glossary](./docs/glossary.md):** Terminology and definitions used throughout the Fiatlux Benchmark

### Submission

* **[Submission Guidelines](./docs/submission.md):** How to package and submit your final model.

---


## Support and Resources

- **Discussions**: Engage in conversations and ask questions about the challenge on [Open Robotics Discourse](https://discourse.openrobotics.org/c/competitions/ai-for-industry-challenge/). The community is encouraged to participate in discussions and assist each other.
- **Issues**: Report any bugs or technical issues via [GitHub Issues](https://github.com/intrinsic-dev/fiatlux/issues). Please refrain from using the Issue tracker for general questions about the challenge.
  - **Note:**: Review the list of [known issues](https://github.com/intrinsic-dev/fiatlux/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22known%20issue%22) and [bugs](https://github.com/intrinsic-dev/fiatlux/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug) before opening a new ticket.
- **Event Page**: Visit the [Fiatlux Benchmark](https://www.intrinsic.ai/events/ai-for-industry-challenge) for official updates.

---

## License

This project is licensed under the Apache License 2.0 - see the individual package files for details.
The [fiatlux_isaac](./fiatlux_utils/fiatlux_isaac/) folder contains files licensed under BSD-3 - see [fiatlux_isaac/LICENSE](./fiatlux_utils/fiatlux_isaac/LICENSE).
