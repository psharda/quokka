find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${amrex_INCLUDE_DIRS_RET})
include_directories(${fmt_INCLUDE_DIRS_RET})

link_libraries(${PYTHON_LIBRARY})
link_libraries(AMReX::amrex)
link_libraries(fmt::fmt)

add_executable(test_advection linear_advection.cpp hyperbolic_system.cpp simulation.cpp test_advection.cpp)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_advection)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

include(CTest)
message(DEBUG "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
add_test(NAME ScalarAdvection COMMAND test_advection WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
