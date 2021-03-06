cmake_minimum_required(VERSION 2.8.3)
project(oa_of_sim)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs message_generation geometry_msgs sensor_msgs)
find_package( OpenCV REQUIRED )

add_message_files(FILES MsgOAOF.msg)

generate_messages(DEPENDENCIES std_msgs geometry_msgs)

catkin_package(
  LIBRARIES oa_of_sim
  CATKIN_DEPENDS roscpp std_msgs cv_bridge image_transport geometry_msgs mav_msgs sensor_msgs
  DEPENDS system_lib
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(oa_of_sim_no_setpoint_001 src/oa_of_sim_no_setpoint_001.cpp)
target_link_libraries(oa_of_sim_no_setpoint_001 ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(oa_of_sim_no_setpoint_001 oa_of_sim_generate_messages_cpp)

add_executable(oa_of_sim_no_setpoint_011 src/oa_of_sim_no_setpoint_011.cpp)
target_link_libraries(oa_of_sim_no_setpoint_011 ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(oa_of_sim_no_setpoint_011 oa_of_sim_generate_messages_cpp)

add_executable(oa_of_sim_no_setpoint_012 src/oa_of_sim_no_setpoint_012.cpp)
target_link_libraries(oa_of_sim_no_setpoint_012 ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(oa_of_sim_no_setpoint_012 oa_of_sim_generate_messages_cpp)

add_executable(oa_of_sim_no_setpoint_041 src/oa_of_sim_no_setpoint_041.cpp)
target_link_libraries(oa_of_sim_no_setpoint_041 ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(oa_of_sim_no_setpoint_041 oa_of_sim_generate_messages_cpp)

add_executable(oa_of_sim_setpoint_001 src/oa_of_sim_setpoint_001.cpp)
target_link_libraries(oa_of_sim_setpoint_001 ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(oa_of_sim_setpoint_001 oa_of_sim_generate_messages_cpp)

add_executable(oa_of_sim_no_setpoint_rr_001 src/oa_of_sim_no_setpoint_rr_001.cpp)
target_link_libraries(oa_of_sim_no_setpoint_rr_001 ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(oa_of_sim_no_setpoint_rr_001 oa_of_sim_generate_messages_cpp)

add_executable(oa_of_sim_yaw_eff src/oa_of_sim_yaw_eff.cpp)
target_link_libraries(oa_of_sim_yaw_eff ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(oa_of_sim_yaw_eff oa_of_sim_generate_messages_cpp)
