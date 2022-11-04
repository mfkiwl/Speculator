SET(TEST_RESSOURCES_DIR ${CMAKE_BINARY_DIR}/test_ressources)

    MACRO(unit_test class)
        ADD_EXECUTABLE(${class}_test ${class}_test.cpp) #File_Comparator.cpp
        TARGET_LINK_LIBRARIES(${class}_test gtest gtest_main fasst ${FFTW_LIB} ${SNDFILE_LIB} tinyxml2)
        SET_TARGET_PROPERTIES (${class}_test PROPERTIES FOLDER Tests_functionnal)
           
        FILE(COPY "ressources/" DESTINATION ${TEST_RESSOURCES_DIR})
       
        SET_SOURCE_FILES_PROPERTIES (${class}_test.cpp
        PROPERTIES COMPILE_DEFINITIONS "INPUT_DATA_DIR=\"${CMAKE_BINARY_DIR}/test_ressources\";BIN_DIR=\"${CMAKE_BINARY_DIR}/bin\"")
        
        
    ENDMACRO()

    unit_test(comp-rx)
    unit_test(model-estimation)
    unit_test(source-estimation)
    
    # Comp-rx test on example 1,2,3 with 0.1% relative error margin (0% does not pass on CI)
    ADD_TEST(NAME comp-rx_test_example1 COMMAND comp-rx_test "ex1" "0.1")
    ADD_TEST(NAME comp-rx_test_example2 COMMAND comp-rx_test "ex2" "0.1")
    ADD_TEST(NAME comp-rx_test_example3 COMMAND comp-rx_test "ex3" "0.1")
    
    # model-estimation test on example 1,2,3 with 0% relative error margin 
    ADD_TEST(NAME model-estimation_test_example1 COMMAND model-estimation_test "ex1" "0.1")
    ADD_TEST(NAME model-estimation_test_example2 COMMAND model-estimation_test "ex2" "0.1")
    ADD_TEST(NAME model-estimation_test_example3 COMMAND model-estimation_test "ex3" "0.1")
    
    # source-estimation test on example 1,2,3 with 0% relative error margin     
    ADD_TEST(NAME source-estimation_test_example1 COMMAND source-estimation_test "ex1" "0")
    ADD_TEST(NAME source-estimation_test_example2 COMMAND source-estimation_test "ex2" "0")
    ADD_TEST(NAME source-estimation_test_example3 COMMAND source-estimation_test "ex3" "0")
    