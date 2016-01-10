add_definitions(-Dclew_STATIC -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DGLEW_STATIC)

add_subdirectory(external)

include_directories(include external/clew/include external/glew/include)

add_library(pbdgpu STATIC
	include/util/gpu_mem_allocator.hpp
        src/util/gpu_mem_allocator.cpp
	include/util/gl_buffer_allocator.hpp
	src/util/gl_buffer_allocator.cpp
        include/util/cl_buffer_allocator.hpp
        src/util/cl_buffer_allocator.cpp

        include/kernels.hpp
        src/kernels.cpp
        cmake/kernels.cpp.in
        include/kernelInclude/particle.h

        src/kernelSrc/prediction.cl
	
	include/util/functions.hpp
	src/util/functions.cpp
	)

file(GLOB KERNEL_HEADERS include/kernelInclude/*.h)
file(GLOB KERNEL_SOURCES src/kernelSrc/*.cl)

add_custom_command(
		OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cpp
		COMMAND cmake -DDIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_LIST_DIR}/cmake/processKernels.cmake
		DEPENDS ${KERNEL_HEADERS} ${KERNEL_SOURCES} cmake/kernels.cpp.in include/kernels.hpp)

add_custom_target(resources DEPENDS ${KERNEL_SOURCES} ${KERNEL_HEADERS} cmake/kernels.cpp.in include/kernels.hpp)

add_dependencies(pbdgpu resources)

find_package(OpenGL)

target_link_libraries(pbdgpu ${OPENGL_gl_LIBRARY} glew clew)

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(
	doc
	ALL 
	${DOXYGEN_EXECUTABLE}
	${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY
	${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Generating API documentation with Doxygen" VERBATIM
	)
endif(DOXYGEN_FOUND)
