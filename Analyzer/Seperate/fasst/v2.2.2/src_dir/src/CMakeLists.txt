# Build tinyxml2/
ADD_SUBDIRECTORY(tinyxml2)

# Build libfasst
ADD_SUBDIRECTORY(fasst)

IF(TEST)
ADD_SUBDIRECTORY(fasst_test)
ENDIF()
# Build the three main executables
INCLUDE_DIRECTORIES("tinyxml2/")

ADD_EXECUTABLE(comp-rx comp-rx.cpp)
TARGET_LINK_LIBRARIES(comp-rx fasst ${FFTW_LIB} ${SNDFILE_LIB} tinyxml2)
ADD_EXECUTABLE(model-estimation model-estimation.cpp)
TARGET_LINK_LIBRARIES (model-estimation fasst ${FFTW_LIB} ${SNDFILE_LIB} tinyxml2)
ADD_EXECUTABLE(source-estimation source-estimation.cpp)
TARGET_LINK_LIBRARIES(source-estimation fasst ${FFTW_LIB} ${SNDFILE_LIB} tinyxml2)

# Copy necessary extern dll to binary directories

MACRO(copy_libsndfile_dll project_name)

    ADD_CUSTOM_COMMAND(TARGET ${project_name} POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SNDFILE_DLL}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}/libsndfile-1.dll)
        
    ADD_CUSTOM_COMMAND(TARGET ${project_name} POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SNDFILE_DLL}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}/libsndfile-1.dll)

ENDMACRO()

MACRO(copy_fftw_dll project_name)

    ADD_CUSTOM_COMMAND(TARGET ${project_name} POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${FFTW_DLL} 
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}/libfftw3-3.dll)
        
    ADD_CUSTOM_COMMAND(TARGET ${project_name} POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${FFTW_DLL} 
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}/libfftw3-3.dll)
        
ENDMACRO()

IF(MSVC)

    copy_libsndfile_dll(comp-rx)
    copy_libsndfile_dll(model-estimation)
    copy_libsndfile_dll(source-estimation)
  
    if(EIGEN_FFTW_DEFAULT)
        copy_fftw_dll(comp-rx)
        copy_fftw_dll(model-estimation)
        copy_fftw_dll(source-estimation)
    endif()
    
    # Add executables and dll to Windows installer
    INSTALL(TARGETS comp-rx model-estimation source-estimation
        RUNTIME DESTINATION bin)
    INSTALL(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/
        DESTINATION bin/
        FILES_MATCHING PATTERN "*.dll")
ENDIF()

# For os x : add binaries to bin root folder
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # Manage binaries path for install command
    INSTALL(TARGETS comp-rx model-estimation source-estimation
            RUNTIME DESTINATION bin)
ENDIF()

IF(PROFILING)

    ADD_EXECUTABLE(comp-rx_profile comp-rx_profile.cpp)
    TARGET_LINK_LIBRARIES(comp-rx_profile fasst ${FFTW_LIB} ${SNDFILE_LIB} tinyxml2)
    ADD_EXECUTABLE(model-estimation_profile model-estimation_profile.cpp)
    TARGET_LINK_LIBRARIES (model-estimation_profile fasst ${FFTW_LIB} ${SNDFILE_LIB} tinyxml2)
    ADD_EXECUTABLE(source-estimation_profile source-estimation_profile.cpp)
    TARGET_LINK_LIBRARIES(source-estimation_profile fasst ${FFTW_LIB} ${SNDFILE_LIB} tinyxml2)
    
    IF(MSVC)
    
        copy_libsndfile_dll(comp-rx_profile)
        copy_libsndfile_dll(model-estimation_profile)
        copy_libsndfile_dll(source-estimation_profile)
        
        IF(EIGEN_FFTW_DEFAULT)
            copy_fftw_dll(comp-rx_profile)
            copy_fftw_dll(model-estimation_profile)
            copy_fftw_dll(source-estimation_profile)
        ENDIF()
        
        # Add executables and dll to Windows installer
        INSTALL(TARGETS comp-rx_profile model-estimation_profile source-estimation_profile
            RUNTIME DESTINATION bin)
 
    ENDIF()
    
ENDIF()
