# Run on real car 
___

# 1. Run lidar node 

go to dir: 
`cd ~/ros2_ws`

download packages: 
`rosdep update`
`rosdep install --from-paths src --ignore-src -r -y`

build: 
`cd ./src`
`colcon build --symlink-install`

source:
`source install/setup.bash`

give permission to usb:
`./src/sllidar_ros2/scripts/create_udev_rules.sh`
(if it desn't work: `sudo chmod 666 /dev/ttyUSB0`)

run node:
`ros2 launch sllidar_ros2 sllidar_a2m8_launch.py`

___

# 2. Run Main Controll Node
