project(PBDGPU)

cmake_minimum_required(VERSION 3.1)

enable_testing()

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(PBDGPU)
add_subdirectory(tests)
add_subdirectory(demos)

