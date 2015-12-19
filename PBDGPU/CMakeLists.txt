add_definitions(-Dclew_STATIC -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DGLEW_STATIC)

add_subdirectory(external)

include_directories(include external/clew/include external/glew/include)

add_library(pbdgpu STATIC
	include/util/gpu_mem_allocator.hpp
	include/util/gl_buffer_allocator.hpp
	src/util/gl_buffer_allocator.cpp
	
	include/util/functions.hpp
	src/util/functions.cpp
	)

target_link_libraries(pbdgpu glew clew)