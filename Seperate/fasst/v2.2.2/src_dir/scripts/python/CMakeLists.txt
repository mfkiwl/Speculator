# Configure python module
SET(FASST_EXECUTABLE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
CONFIGURE_FILE(fasst.py.in ${CMAKE_CURRENT_BINARY_DIR}/fasst.py @ONLY)

IF(TEST)
	IF(WIN32)
	    ADD_TEST(NAME python_test COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/fasst.py -v WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
	ELSE()
    ADD_TEST(python_test
        python fasst.py -v)
		ENDIF()
ENDIF()

# Add to Windows or os x installer
IF(MSVC OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # Reconfigure for package
    SET(FASST_EXECUTABLE_DIR "${CMAKE_INSTALL_PREFIX}/bin")
    CONFIGURE_FILE(fasst.py.in ${CMAKE_BINARY_DIR}/package/system/scripts/python/fasst.py @ONLY)

    INSTALL(FILES ${CMAKE_BINARY_DIR}/package/system/scripts/python/fasst.py
        DESTINATION scripts/python/)
ENDIF()
