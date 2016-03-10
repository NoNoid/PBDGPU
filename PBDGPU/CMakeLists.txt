add_definitions(-Dclew_STATIC -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DGLEW_STATIC)

add_subdirectory(external)

include_directories(include external/clew/include external/glew/include)

set(KERNEL_HEADERS include/kernelInclude/particle.h include/kernelInclude/distanceConstraintData.h)
set(KERNEL_SOURCES src/kernelSrc/prediction.cl src/kernelSrc/update.cl src/kernelSrc/planeCollision.cl src/kernelSrc/distanceConstraint.cl)

add_library(pbdgpu STATIC
        ${KERNEL_HEADERS}
        ${KERNEL_SOURCES}
        include/util/gpu_mem_allocator.hpp
        src/util/gpu_mem_allocator.cpp
        include/util/gl_buffer_allocator.hpp
        src/util/gl_buffer_allocator.cpp
        include/util/cl_buffer_allocator.hpp
        src/util/cl_buffer_allocator.cpp
		include/util/gl_copy_buffer_allocator.hpp
		src/util/gl_copy_buffer_allocator.cpp

        include/kernels.hpp
        src/kernels.cpp
        cmake/kernels.cpp.in

        include/util/functions.hpp
        src/util/functions.cpp

        include/simulation_data.hpp
        src/simulation_data.cpp

        include/constraint.hpp
        src/constraint.cpp
        include/constraints/common_constraint.hpp
        src/constraints/common_constraint.cpp
        include/constraints/plane_collision_constraint.hpp
        src/constraints/plane_collision_constraint.cpp include/constraints/distance_constraint.hpp src/constraints/distance_constraint.cpp)

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
