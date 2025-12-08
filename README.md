# TODO
___

1. Teleportacja auta podczas resetu
- dodać bridge: `/gazebo/set_entity_state@gazebo_msgs/srv/SetEntityState`
- sprawdzić dostępność usługi przez wywołanie w terminalu: `ros2 service list | grep set_entity_state`
- teleportowac auto z terminala: `ros2 service call /gazebo/set_entity_state gazebo_msgs/srv/SetEntityState "{state: {name: \"vehicle_blue\", pose: {position: {x: -6.65, y: 14.61, z: 0.05}}}}"`
- jak zadziała to powinno zadziałać to co jest już zaimoplementowane
# Resolved 
___
1. render() - symulacja zgłasza warning że nie podano render_mode = True, ale można to zignorować - zaimplementowałem funkcję tak że tego nie potrzebuje. Przyszłościowo można dać ten render_mode jako argument i nic z nim nie ribić żeby nie krzyczało 

