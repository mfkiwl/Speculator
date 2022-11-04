# Configure example script

SET(FASST_MATLAB_DIR ${CMAKE_BINARY_DIR}/scripts/MATLAB)

message("PATH : " $ENV{PATH})

# Looking for bss_eval backend
FIND_PATH(BSS_EVAL_PATH bss_eval_images.m PATHS $ENV{PATH})
IF("${BSS_EVAL_PATH}" STREQUAL "BSS_EVAL_PATH-NOTFOUND")
    MESSAGE(STATUS "[Example 3] : BSS-EVAL backend disabled")
    set(BSS_EVAL_PATH "")
    set(BSS_EVAL_BACK_END 0)
ELSE()
    MESSAGE(STATUS "[Example 3] : BSS-EVAL backend enabled")
    set(BSS_EVAL_BACK_END 1)
ENDIF()

# Looking for bss_eval backend
FIND_PATH(MBSS_LOCATE_PATH MBSS_locate_spec.m PATHS $ENV{PATH})
IF("${MBSS_LOCATE_PATH}" STREQUAL "MBSS_LOCATE_PATH-NOTFOUND")
    MESSAGE(STATUS "[Example 3] : MBSS-LOCATE backend disabled")
    set(MBSS_LOCATE_PATH "")
    set(MBSS_LOCATE_BACK_END 0)
ELSE()
    MESSAGE(STATUS "[Example 3] : MBSS-LOCATE backend enabled")
    set(MBSS_LOCATE_BACK_END 1)
ENDIF()

# Looking for peass back end
FIND_PATH(PEASS_PATH PEASS_ObjectiveMeasure.m PATHS $ENV{PATH})
IF("${PEASS_PATH}" STREQUAL "PEASS_PATH-NOTFOUND")
    MESSAGE(STATUS "[Example 3] : PEASS backend disabled")
    set(PEASS_PATH "")
    set(PEASS_BACk_END 0)
ELSE()
    MESSAGE(STATUS "[Example 3] : PEASS backend enabled")
    set(PEASS_BACk_END 1)
ENDIF()

CONFIGURE_FILE(example3_old.m.in ${CMAKE_CURRENT_BINARY_DIR}/example3_old.m @ONLY)
CONFIGURE_FILE(male1.wav ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
CONFIGURE_FILE(male2.wav ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Add to Windows installer
IF(MSVC)
    # directory for scripts MATLAB under Program Files
    SET(FASST_MATLAB_DIR "${CMAKE_INSTALL_PREFIX}/scripts/MATLAB")

    CONFIGURE_FILE(example3_old.m.in ${CMAKE_BINARY_DIR}/examples/example3_old/package/example3_old.m @ONLY)
    CONFIGURE_FILE(male1.wav ${CMAKE_BINARY_DIR}/examples/example3_old/package/ COPYONLY)
    CONFIGURE_FILE(male2.wav ${CMAKE_BINARY_DIR}/examples/example3_old/package/ COPYONLY)

    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/package/example3_old.m DESTINATION examples/example3_old/)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/package/male1.wav DESTINATION examples/example3_old/)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/package/male2.wav DESTINATION examples/example3_old/)
ENDIF()
