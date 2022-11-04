# Configure example script

SET(FASST_MATLAB_DIR ${CMAKE_BINARY_DIR}/scripts/MATLAB)

# Looking for peass back end
FIND_PATH(PEASS_PATH PEASS_ObjectiveMeasure.m PATHS $ENV{PATH})
IF("${PEASS_PATH}" STREQUAL "PEASS_PATH-NOTFOUND")
    MESSAGE(STATUS "[Example 2] : PEASS backend disabled")
    SET(PEASS_PATH "")
    SET(PEASS_BACK_END 0)
ELSE()
    MESSAGE(STATUS "[Example 2] : PEASS backend enabled")
    set(PEASS_BACK_END 1)
ENDIF()

CONFIGURE_FILE(example2.m.in ${CMAKE_CURRENT_BINARY_DIR}/example2.m @ONLY)
CONFIGURE_FILE(src_imag_1_anechoic.wav ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
CONFIGURE_FILE(src_imag_2_anechoic.wav ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
CONFIGURE_FILE(LICENSE.txt ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Add to windows or os x installer and other installation process
IF(MSVC OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # directory for scripts MATLAB under Program Files
    SET(FASST_MATLAB_DIR "${CMAKE_INSTALL_PREFIX}/scripts/MATLAB")
    SET(PEASS_PATH "")
    SET(PEASS_BACK_END 0)
    CONFIGURE_FILE(example2.m.in ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example2/example2.m @ONLY)
    CONFIGURE_FILE(src_imag_1_anechoic.wav ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example2/ COPYONLY)
    CONFIGURE_FILE(src_imag_2_anechoic.wav ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example2/ COPYONLY)
    CONFIGURE_FILE(LICENSE.txt ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example2/ COPYONLY)

    IF(MSVC)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example2/example2.m DESTINATION examples/example2/)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example2/src_imag_1_anechoic.wav DESTINATION examples/example2/)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example2/src_imag_2_anechoic.wav DESTINATION examples/example2/)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/example2/LICENSE.txt DESTINATION examples/example2/)
    ENDIF()
ENDIF()
