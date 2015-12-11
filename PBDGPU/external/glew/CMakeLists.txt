add_definitions(-DGLEW_STATIC)

include_directories(include)

add_library(glew STATIC src/glew.c)