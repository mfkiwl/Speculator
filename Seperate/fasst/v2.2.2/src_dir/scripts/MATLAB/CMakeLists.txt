# Configure 3 main functions
SET(FASST_EXECUTABLE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
FOREACH(FILE_TO_CONFIGURE
        fasst_compute_mixture_covariance_matrix.m
        fasst_estimate_source_parameters.m
        fasst_estimate_sources.m)
    CONFIGURE_FILE(${FILE_TO_CONFIGURE}.in ${CMAKE_CURRENT_BINARY_DIR}/${FILE_TO_CONFIGURE} @ONLY)
ENDFOREACH()

# Copy other helper functions (all *.m files)
FILE(GLOB FILES_TO_COPY *.m)
FOREACH(FILE_TO_COPY ${FILES_TO_COPY})
    CONFIGURE_FILE(${FILE_TO_COPY} ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
ENDFOREACH()

# Add to Windows or os x installer
IF(MSVC OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # Reconfigure for package : give .m file directory of binaries
    SET(FASST_EXECUTABLE_DIR "${CMAKE_INSTALL_PREFIX}/bin")
    FOREACH(FILE_TO_CONFIGURE
            fasst_compute_mixture_covariance_matrix.m
            fasst_estimate_source_parameters.m
            fasst_estimate_sources.m)

        CONFIGURE_FILE(${FILE_TO_CONFIGURE}.in ${CMAKE_BINARY_DIR}/package/system/scripts/MATLAB/${FILE_TO_CONFIGURE} @ONLY)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/system/scripts/MATLAB/${FILE_TO_CONFIGURE}
            DESTINATION scripts/MATLAB/)

    ENDFOREACH()

    FOREACH(FILE_TO_CONFIGURE
            fasst_loadRx.m
            fasst_loadXML.m
            fasst_writeRx.m
            fasst_writeXML.m)

        CONFIGURE_FILE(${FILE_TO_CONFIGURE} ${CMAKE_BINARY_DIR}/package/system/scripts/MATLAB/${FILE_TO_CONFIGURE} @ONLY)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/package/system/scripts/MATLAB/${FILE_TO_CONFIGURE}
            DESTINATION scripts/MATLAB/)

    ENDFOREACH()

    #INSTALL(FILES .
    #    DESTINATION ${CMAKE_BINARY_DIR}/package/system/scripts/MATLAB/
    #    FILES_MATCHING PATTERN "*.m")
ENDIF()
