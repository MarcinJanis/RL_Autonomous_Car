#!/bin/bash

colcon build --packages-select autonomous_vehicles

source install/setup.bash

ros2 launch autonomous_vehicles RL.auto.launch.py