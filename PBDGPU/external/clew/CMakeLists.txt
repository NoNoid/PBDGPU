if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_1_APIS -Dclew_STATIC)

include_directories(include)

add_subdirectory(src)

