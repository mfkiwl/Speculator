# Configure example script

SET(FASST_MATLAB_DIR ${CMAKE_BINARY_DIR}/scripts/MATLAB)
CONFIGURE_FILE(example3.m.in ${CMAKE_CURRENT_BINARY_DIR}/example3.m @ONLY)
SET(FASST_PYTHON_DIR ${CMAKE_BINARY_DIR}/scripts/python)
CONFIGURE_FILE(example3.py.in ${CMAKE_CURRENT_BINARY_DIR}/example3.py @ONLY)
CONFIGURE_FILE(input.wav ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
CONFIGURE_FILE(LICENSE.txt ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Add to windows or os x installer and other installation process
IF(MSVC OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # directory for scripts MATLAB under Program Files
    SET(FASST_MATLAB_DIR "${CMAKE_INSTALL_PREFIX}/scripts/MATLAB")
    CONFIGURE_FILE(example3.m.in ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example3/example3.m @ONLY)
    SET(FASST_PYTHON_DIR "${CMAKE_INSTALL_PREFIX}/scripts/python")
    CONFIGURE_FILE(example3.py.in ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example3/example3.py @ONLY)
    CONFIGURE_FILE(input.wav ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example3 COPYONLY)
    CONFIGURE_FILE(LICENSE.txt ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example3 COPYONLY)

    IF(MSVC)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example3/example3.m DESTINATION examples/example3/)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example3/example3.py DESTINATION examples/example3/)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example3/input.wav DESTINATION examples/example3/)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example3/LICENSE.txt DESTINATION examples/example3/)
    ENDIF()
ENDIF()
