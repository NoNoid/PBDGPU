add_definitions(-Dclew_STATIC -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DGLEW_STATIC)

add_subdirectory(external)

include_directories(include)

add_library(pbdgpu STATIC
	include/Util/gpu_mem_allocator.hpp
	src/Util/gpu_mem_allocator.cpp
	)

target_link_libraries(pbdgpu glew clew)