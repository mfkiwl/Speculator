    # temporary test for development purpose
    SET_SOURCE_FILES_PROPERTIES(ERBLETRepr_port.cpp
        PROPERTIES COMPILE_DEFINITIONS INPUT_DATA_DIR_ERBLET=\"${CMAKE_BINARY_DIR}/../../test/ERB/audioFiles\")

    MACRO(unit_test class)
        ADD_EXECUTABLE(${class}_port ${class}_port.cpp)
        TARGET_LINK_LIBRARIES(${class}_port gtest gtest_main fasst ${FFTW_LIB} ${SNDFILE_LIB} tinyxml2)
        SET_TARGET_PROPERTIES (${class}_port PROPERTIES FOLDER Tests_porting)
        ADD_TEST(NAME ${class}_test COMMAND ${class}_port)
    ENDMACRO()

    unit_test(ERBLETRepr)

