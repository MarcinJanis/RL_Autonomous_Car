# Reinforcment learning - autonomous mecanum car 

## **Project Overview**

This project implements an end-to-end Reinforcement Learning (RL) pipeline for an autonomous car equipped with Mecanum wheels. The agent is trained in a simulated Gazebo environment to follow roads at high speeds while avoiding obstacles, using a fused sensor input (Lidar + RGB Camera). The system is designed with a "Sim-to-Real" approach, allowing deployment on physical hardware via NVIDIA Jetson Xavier.


___
## **System architecture**

Car is configure to operate based on sensors:
- **2D lidar** [Slamtec A2M8](https://bucket-download.slamtec.com/20b2e974dd7c381e46c78db374772e31ea74370d/LD208_SLAMTEC_rplidar_datasheet_A2M8_v2.6_en.pdf)

- **RGB Camera** [Realsens D435i](https://www.realsenseai.com/products/depth-camera-d435i/)

___
**Neural Network Model**

The core logic resides in: <br>`RL_Autonomous_Car\siapwpa_ros2_project-main_rl\autonomous_vehicles\autonomous_vehicles\net_agent\net_v1.py`. 

Model was implemented in PyTorch.

**Inputs**:
- Lidar Data: 1D Laser scan arrays for proximity detection.
- Visual Data: RGB frames for road following and lane detection.

**Output:**
- Continuous actions for longitudinal, lateral, and angular velocity.

Model was trained using **PPO** algorythm implemented in [**stable-baseline-3**](https://stable-baselines3.readthedocs.io/en/master/)



- Car with mecanum wheels, sensors and simulation envirnment is configure in: <br>
`RL_Autonomous_Car\siapwpa_ros2_project-main_rl\models\walls\mecanum.sdf`

- Environment to reinforcement learing: <br>
`RL_Autonomous_Car\siapwpa_ros2_project-main_rl\autonomous_vehicles\autonomous_vehicles\training\gazebo_car_env.py`

- Training script with traning details: <br>
`RL_Autonomous_Car\siapwpa_ros2_project-main_rl\autonomous_vehicles\autonomous_vehicles\training\training_SB3.py`

Model purpose was to **follow the road**, **maximizie speed** and **avoid collision**.

___
## **Results**

**Visualisation**

///////ad here gifs - recoders from gazebo

///////ad here gifs - records of visualisation

Note: Colour of line drawed by car is heatmap of speed gradient in order to better representation of working. 


### How to run inference:

Open folder `siapwpa_ros2_project-main_rl/` in docker container 

Perform following commands:

```bash
colcon build --packages-select autonomous_vehicles

source install/setup.bash`

ros2 launch autonomous_vehicles sim_inference.auto.launch.py`
```

### How to run training:

Open folder `siapwpa_ros2_project-main_rl/` in docker container 

Perform following commands:

```bash
colcon build --packages-select autonomous_vehicles

source install/setup.bash

ros2 launch autonomous_vehicles RL.auto.launch.py
```
___

## **Hardware Deployment (Sim-to-Real)**
The system has been fully adapted to transition from simulation to a physical environment. The trained RL policy is deployed on a Mecanum car capable of omnidirectional movement.

### **Hardware Specification**:
Compute: [**NVIDIA Jetson Xavier NX 2**](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-nx/) 

Sensors: 
- **2D lidar** [Slamtec A2M8](https://bucket-download.slamtec.com/20b2e974dd7c381e46c78db374772e31ea74370d/LD208_SLAMTEC_rplidar_datasheet_A2M8_v2.6_en.pdf)
- **RGB Camera** [Realsens D435i](https://www.realsenseai.com/products/depth-camera-d435i/)

Middleware: [**ROS2 Foxy**](https://docs.ros.org/en/foxy/index.html).

Software Integration
The hardware-specific implementation, including ROS2 nodes sensor bridges and communication with motor controllers, is located in: RL_Autonomous_Car/siapwpa_ros2_project-main_rl/real_car_controller

### How to deploy control system on hardware:

To run, perform following commands



## **Directory Structure**

`models/`: SDF and mesh files for the car and environment.

`autonomous_vehicles/autonomous_vehicles/` <br>

`training/`: Custom OpenAI Gym/Gymnasium wrapper for the Gazebo environment and training script.

`net_agent/`: PyTorch implementation of the neural network.\

`car_controller`: ROS2 nodes and scripts for inference

`net_road_segmentation`: Files for pre-training encoder, used in neural network

`real_car_controller/`: ROS2 nodes for hardware deployment.

<!-- 
RL_Autonomous_Car/
├── models/                       # SDF/Mesh files for the Mecanum car and Gazebo worlds
├── net_road_segmentation/        # Scripts for pre-training the visual encoder (CNN)
└── siapwpa_ros2_project-main_rl/ # Main ROS2 Workspace
    └── autonomous_vehicles/
        └── autonomous_vehicles/
            ├── net_agent/        # PyTorch PPO architecture (net_v1.py)
            ├── training/         # Gym/Gazebo environment & training scripts (SB3)
            ├── car_controller/   # Inference nodes and logic for simulation
            └── real_car_controller/ # ROS2 hardware drivers & Sim-to-Real bridge -->
