add_definitions(-Dclew_STATIC -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DGLEW_STATIC)

find_package(GLUT REQUIRED)
find_package(GLM REQUIRED)

include_directories(${GLM_INCLUDE_DIRS} ${GLUT_INCLUDE_DIR} ../PBDGPU/external/clew/include ../PBDGPU/external/glew/include ../PBDGPU/include)

set(SHADERS
    shaders/particle.vert
    shaders/particle.frag
    shaders/plane.vert
    shaders/plane.frag)

file(COPY ${SHADERS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shaders)

add_executable(simplePBD simplePBD.cpp ${SHADERS})
target_link_libraries(simplePBD ${GLUT_LIBRARIES} glew clew pbdgpu)
