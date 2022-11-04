# Configure example script
SET(FASST_MATLAB_DIR ${CMAKE_BINARY_DIR}/scripts/MATLAB)
CONFIGURE_FILE(example1.m.in ${CMAKE_CURRENT_BINARY_DIR}/example1.m @ONLY)
SET(FASST_PYTHON_DIR ${CMAKE_BINARY_DIR}/scripts/python)
CONFIGURE_FILE(example1.py.in ${CMAKE_CURRENT_BINARY_DIR}/example1.py @ONLY)

# Copy input data
CONFIGURE_FILE(Shannon_Hurley__Sunrise__inst__mix.wav ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
CONFIGURE_FILE(Shannon_Hurley__Sunrise__inst__license.txt ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Add to windows or os x installer and other installation process
IF(MSVC OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # directory for scripts MATLAB under Program Files
    SET(FASST_MATLAB_DIR "${CMAKE_INSTALL_PREFIX}/scripts/MATLAB")
    CONFIGURE_FILE(example1.m.in ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example1/example1.m @ONLY)

    SET(FASST_PYTHON_DIR "${CMAKE_INSTALL_PREFIX}/scripts/python")
    CONFIGURE_FILE(example1.py.in ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example1/example1.py @ONLY)

    FILE(COPY Shannon_Hurley__Sunrise__inst__mix.wav DESTINATION ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example1/)
    FILE(COPY Shannon_Hurley__Sunrise__inst__license.txt DESTINATION ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example1/)

    IF(MSVC)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example1/example1.m DESTINATION examples/example1/)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example1/example1.py DESTINATION examples/example1/)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example1/Shannon_Hurley__Sunrise__inst__mix.wav DESTINATION examples/example1/)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example1/Shannon_Hurley__Sunrise__inst__license.txt DESTINATION examples/example1/)
    ENDIF()


ENDIF()
