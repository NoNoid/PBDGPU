add_definitions(-Dclew_STATIC -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DGLEW_STATIC)

find_package(GLUT REQUIRED)

include_directories(${GLUT_INCLUDE_DIR} ../PBDGPU/external/clew/include ../PBDGPU/external/glew/include ../PBDGPU/include)

add_executable(testOpenGLCL testOpenGLCL.cpp)
target_link_libraries(testOpenGLCL ${GLUT_LIBRARIES} glew clew pbdgpu)
add_test(interopTest testOpenGLCL)

add_executable(bufferTest testBuffers.cpp)
target_link_libraries(bufferTest ${GLUT_LIBRARIES} glew clew pbdgpu)
add_test(bufferTest bufferTest)

add_executable(testSort testSort.cpp)
target_link_libraries(testSort ${GLUT_LIBRARIES} glew clew pbdgpu)
add_test(testSort testSort)

if(VALGRIND)
    add_valgrind_test(valgrind_interopTest testOpenGLCL)
    add_valgrind_test(valgrind_bufferTest bufferTest)
endif(VALGRIND)
