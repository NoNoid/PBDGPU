add_library( clew STATIC clew.c )

if(UNIX)
target_link_libraries( clew dl )
endif(UNIX)
