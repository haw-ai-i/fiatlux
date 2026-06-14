# Fiatlux Project Phase 3 Plan: Sim-to-Real Gap Closure

This document outlines the formalized plan for Phase 3 of the Fiatlux Benchmark. Phase 3 assumes a policy has been pre-trained and validated in simulation (Phase 2) and focuses on the infrastructure, calibration, and driver adapters required to transfer this policy to the physical Unitree G1 humanoid robot.

---

## 1. Physical Environment & Hardware Setup

The physical deployment room contains:
* **Humanoid Robot**: Unitree G1 humanoid robot equipped with two **Inspire dexterous hands**.
* **Scene Objects**: A real industrial/household ladder, a lamp fixture mounted at height, and a screw-in light bulb.
* **Onboard/Offboard Compute**: A local compute station running ROS 2 to run the policy model (CPU/GPU) and communicate with the robot's onboard computer.

---

## 2. Infrastructure Required for Sim-to-Real Transition

To close the sim-to-real gap and safely execute the simulation-trained policy on physical hardware, the following infrastructure must be implemented:

```mermaid
graph LR
    Policy[Trained Policy] --> ControllerAdapter[ROS 2 to Unitree SDK 2 Bridge]
    ControllerAdapter --> RealRobot[Physical G1 & Inspire Hands]
    RealSensors[Cameras & F/T Sensors] --> PerceptionBridge[State Estimation & Alignment]
    PerceptionBridge --> Policy
```

### A. Hardware Driver & Controller Adapter (Policy Action Interface)
* **SDK Bridge**: Implement a ROS 2 hardware interface mapping the policy's action outputs (e.g. Cartesian end-effector targets, joint positions, or joint torques) to the **Unitree SDK 2 / socket API** for G1 legs and arms, and the serial/CAN bus protocol for the **Inspire hands**.
* **Safety Filters**: Implement a low-level safety wrapper running on the G1's onboard computer. This filter will:
  - Enforce maximum joint limits and joint velocity safety margins.
  - Saturation-limit the commanded torques to prevent joint over-stress.
  - Instantly command motor shutdown (E-stop) if contact forces exceed predefined limits.

### B. Perception Alignment & State Estimation (Policy Observation Interface)
* **Object Localization**: Set up vision tracking (e.g., ArUco markers or depth-based segmentation models like YOLO) using the G1's wrist and head-mounted cameras to detect:
  - Relative pose of the ladder rungs.
  - Position and angle of the lamp socket.
  - 3D pose of the light bulb.
* **Coordinate Mapping**: Convert the real-world segmented object coordinates into the identical observation frames defined in the simulation (`fiatlux_task_env_cfg.py`), supplying the policy with the same observations it expects from simulation.

### C. System Identification & Calibration (SysID)
* **Latency Calibration**: Measure the round-trip latency (policy inference -> ROS 2 message transmission -> Unitree SDK network delay -> joint motor response). Inject this delay distribution directly into the Isaac Lab simulation during Phase 2 training.
* **Actuator and Compliance Modeling**: Calibrate joint friction, motor torque constants, and gripper spring/compliance behaviors to match physical responses.
* **Friction Identification**: Calibrate the friction coefficient between the physical G1 foot pads and the ladder rungs.

### D. Safety & Fall Prevention Infrastructure (Physical & Software)
* **Physical Gantry/Harness**: A ceiling-mounted or frame-mounted vertical safety tether/harness to catch the G1 humanoid in the event of a slip, loss of balance, or policy failure during climbing.
* **Emergency Stop (E-Stop)**: Set up a dual-path E-Stop system:
  - *Physical E-Stop*: A wireless hardware remote switch that cuts motor power.
  - *Software E-Stop*: A ROS 2 topic listener that catches high-frequency IMU anomalies (e.g., abrupt orientation changes indicating a fall) and commands joint braking.
