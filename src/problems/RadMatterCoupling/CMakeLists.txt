
add_executable(test_radiation_matter_coupling test_radiation_matter_coupling.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_matter_coupling)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME MatterEnergyExchange COMMAND test_radiation_matter_coupling energyexchange.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
