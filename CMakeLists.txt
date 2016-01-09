project(PBDGPU)

cmake_minimum_required(VERSION 3.1)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

enable_testing()

find_program(VALGRIND valgrind)

if(VALGRIND)
    # --gen-suppressions=all
    set( VALGRIND_OPTIONS "--trace-children=yes --leak-check=full --error-limit=no --suppressions=${CMAKE_SOURCE_DIR}/cmake/PBDGPU.supp")

    function(add_valgrind_test name binary)      
      set(VAL_TMP "${VALGRIND} ${VALGRIND_OPTIONS}")
      separate_arguments(VAL_TMP)
      add_test(${name} ${VAL_TMP} ./${binary} ${ARGN})
      set_tests_properties(${name} PROPERTIES FAIL_REGULAR_EXPRESSION "Invalid read of size; blocks are definitely lost in loss record;Conditional jump or move depends on uninitialised value(s)")
    endfunction(add_valgrind_test)

else(VALGRIND)
    set(VALGRIND OFF)
endif(VALGRIND)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(PBDGPU)
add_subdirectory(tests)
add_subdirectory(demos)

