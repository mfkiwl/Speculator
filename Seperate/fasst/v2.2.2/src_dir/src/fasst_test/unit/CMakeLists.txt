
    # Find input data during tests
    SET_SOURCE_FILES_PROPERTIES(Audio_test.cpp
        PROPERTIES COMPILE_DEFINITIONS INPUT_DATA_DIR=\"${CMAKE_BINARY_DIR}/examples/example1\")
    # temporary test for development purpose
    SET_SOURCE_FILES_PROPERTIES(ERBLETRepr_test.cpp
        PROPERTIES COMPILE_DEFINITIONS INPUT_DATA_DIR_ERBLET=\"${CMAKE_BINARY_DIR}/../../test/ERB/audioFiles\")

    MACRO(unit_test class)
        ADD_EXECUTABLE(${class}_test ${class}_test.cpp)
        TARGET_LINK_LIBRARIES(${class}_test gtest gtest_main fasst ${FFTW_LIB} ${SNDFILE_LIB} tinyxml2)
        SET_TARGET_PROPERTIES (${class}_test PROPERTIES FOLDER Tests_unit)
        ADD_TEST(NAME ${class}_test COMMAND ${class}_test)
    ENDMACRO()

    unit_test(Audio)
    unit_test(STFTRepr)
    unit_test(NonNegMatrix)
    unit_test(MixCovMatrix)
    unit_test(MixingParameter)
    unit_test(Sources)
    unit_test(Source)
    unit_test(XMLDoc)
    unit_test(ERBLETRepr)

