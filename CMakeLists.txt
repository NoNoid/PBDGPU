project(PBDGPU)

cmake_minimum_required(VERSION 2.6)

find_package(OpenGL REQUIRED)

find_package(GLUT REQUIRED)

find_package(GLEW REQUIRED)

find_package(OpenCL REQUIRED)

include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})

add_executable(testOpenGLCL Tests/testOpenGLCL.cpp)

target_link_libraries(testOpenGLCL ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCL_LIBRARIES})