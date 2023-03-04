PROJECT(PSO)
cmake_minimum_required(VERSION 3.5)

message(STATUS "${CMAKE_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# OpenMP
FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# # ROS()
# find_package(catkin REQUIRED COMPONENTS
#   geometry_msgs
#   roscpp
#   rospy
#   sensor_msgs
#   tf
#   jsk_recognition_msgs
# )
# include_directories(${catkin_INCLUDE_DIRS})
# link_directories(${catkin_LIBRARY_DIRS})

# 找PCL库
set(PCL_DIR "/usr/local/share/pcl-1.12")
find_package(PCL 1.12 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 找Boost库
set(PCL_ROOT "/usr/local")
find_package(Boost REQUIRED COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# yaml-cpp
find_package(yaml-cpp REQUIRED)
message(STATUS "${yaml-cpp_CONFIG}")
include_directories(/usr/local/include/yaml-cpp)

# OpenCV
SET(OpenCV_DIR /opt/ros/kinetic/include/opencv-3.3.1-dev)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "${OpenCV_CONFIG}")

# # fmt
# find_package(FMT REQUIRED)


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/include/vtk-9.2)


add_library(pso ${PROJECT_SOURCE_DIR}/src/pso.cpp)
# add_library(auto_calib STATIC ${PROJECT_SOURCE_DIR}/src/auto_calib.cpp)
add_library(initCalib ${PROJECT_SOURCE_DIR}/src/initCalib.cpp)

add_executable(testpso test/testpso.cpp)
target_link_libraries(testpso ${OpenCV_LIBS} yaml-cpp pso)

add_executable(testpcl test/testpcl.cpp)
target_link_libraries(testpcl ${PCL_LIBRARIES} ${Boost_LIBRARIES})

add_executable(main src/main.cpp)
target_link_libraries(main initCalib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${OpenCV_LIBRARIES} yaml-cpp pso ${PCL_LIBRARIES} ${Boost_LIBRARIES})