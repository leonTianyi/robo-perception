cmake_minimum_required(VERSION 3.22)
project(robo_perception LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Each subsystem becomes a library target as it is populated, e.g.:
#   add_subdirectory(core)
#   add_subdirectory(runtime)
#   add_subdirectory(detection)
#   add_subdirectory(odometry)
#   add_subdirectory(apps)
