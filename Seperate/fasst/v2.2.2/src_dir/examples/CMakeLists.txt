ADD_SUBDIRECTORY(example1)
ADD_SUBDIRECTORY(example2)
ADD_SUBDIRECTORY(example3)
ADD_SUBDIRECTORY(tools)

#Add quickstarter to installer
IF(MSVC OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    IF(MSVC)
    	CONFIGURE_FILE(QuickStarter_win.txt.in ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/QuickStarter.txt @ONLY)
        CONFIGURE_FILE(install_peass.bat ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/install_peass.bat COPYONLY)
    	INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/QuickStarter.txt DESTINATION examples//)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/install_peass.bat DESTINATION examples//)
    ENDIF()
    
    IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        CONFIGURE_FILE(install_peass.sh ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/install_peass.sh COPYONLY)
    	CONFIGURE_FILE(QuickStarter_mac.txt.in ${CMAKE_BINARY_DIR}/package/examples/${PROJECT_NAME}_${FASST_VERSION}_examples/QuickStarter.txt @ONLY)
    ENDIF()
    
ENDIF()