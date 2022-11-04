# Configure tools functions

CONFIGURE_FILE(computeSteeringVectorFromDirection.m ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
CONFIGURE_FILE(get_freqScaleParams.m ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
CONFIGURE_FILE(get_freqScaleParams.py ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Add to windows or os x installer and other installation process
IF(MSVC OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

    CONFIGURE_FILE(computeSteeringVectorFromDirection.m ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/tools/computeSteeringVectorFromDirection.m COPYONLY)
    CONFIGURE_FILE(get_freqScaleParams.m ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/tools/get_freqScaleParams.m COPYONLY)
    CONFIGURE_FILE(get_freqScaleParams.py ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/tools/get_freqScaleParams.py COPYONLY)

    IF(MSVC)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/tools/computeSteeringVectorFromDirection.m DESTINATION examples/tools/)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/tools/get_freqScaleParams.m DESTINATION examples/tools/)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/tools/get_freqScaleParams.py DESTINATION examples/tools/)
    ENDIF()
ENDIF()
