if (AMReX_SPACEDIM GREATER_EQUAL 2)
    add_executable(test_quirk test_quirk.cpp ${QuokkaObjSources})
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_quirk)
    endif()

    add_test(NAME HydroQuirk COMMAND test_quirk quirk.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
