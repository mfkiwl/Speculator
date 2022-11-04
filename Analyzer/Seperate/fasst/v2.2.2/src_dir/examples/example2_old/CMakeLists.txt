# Configure example script

SET(FASST_MATLAB_DIR ${CMAKE_BINARY_DIR}/scripts/MATLAB)

# Looking for bss_eval backend
FIND_PATH(BSS_EVAL_PATH bss_eval_images.m PATHS $ENV{PATH})
IF("${BSS_EVAL_PATH}" STREQUAL "BSS_EVAL_PATH-NOTFOUND")
    MESSAGE(STATUS "[Example 2] : BSS-EVAL backend disabled")
    set(BSS_EVAL_PATH "")
    set(BSS_EVAL_BACK_END 0)
ELSE()
    MESSAGE(STATUS "[Example 2] : BSS-EVAL backend enabled")
    set(BSS_EVAL_BACK_END 1)
ENDIF()

# Looking for peass back end
FIND_PATH(PEASS_PATH PEASS_ObjectiveMeasure.m PATHS $ENV{PATH})
IF("${PEASS_PATH}" STREQUAL "PEASS_PATH-NOTFOUND")
    MESSAGE(STATUS "[Example 2] : PEASS backend disabled")
    set(PEASS_PATH "")
    set(PEASS_BACk_END 0)
ELSE()
    MESSAGE(STATUS "[Example 2] : PEASS backend enabled")
    set(PEASS_BACk_END 1)
ENDIF()

CONFIGURE_FILE(example2_old.m.in ${CMAKE_CURRENT_BINARY_DIR}/example2_old.m @ONLY)
CONFIGURE_FILE(src_imag_1.wav ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
CONFIGURE_FILE(src_imag_2.wav ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
CONFIGURE_FILE(male_s1.wav ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
CONFIGURE_FILE(female_s1.wav ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
CONFIGURE_FILE(LICENCE.txt ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Add to Windows installer
IF(MSVC)
    # directory for scripts MATLAB under Program Files
    SET(FASST_MATLAB_DIR "${CMAKE_INSTALL_PREFIX}/scripts/MATLAB")

    CONFIGURE_FILE(example2_old.m.in ${CMAKE_BINARY_DIR}/examples/example2_old/package/example2_old.m @ONLY)
    CONFIGURE_FILE(src_imag_1.wav ${CMAKE_BINARY_DIR}/examples/example2_old/package/ COPYONLY)
    CONFIGURE_FILE(src_imag_2.wav ${CMAKE_BINARY_DIR}/examples/example2_old/package/ COPYONLY)
    CONFIGURE_FILE(male_s1.wav ${CMAKE_BINARY_DIR}/examples/example2_old/package/ COPYONLY)
    CONFIGURE_FILE(female_s1.wav ${CMAKE_BINARY_DIR}/examples/example2_old/package/ COPYONLY)
    CONFIGURE_FILE(LICENCE.txt ${CMAKE_BINARY_DIR}/examples/example2_old/package/ COPYONLY)
    
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/package/example2_old.m DESTINATION examples/example2_old/)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/package/src_imag_1.wav DESTINATION examples/example2_old/)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/package/src_imag_2.wav DESTINATION examples/example2_old/)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/package/male_s1.wav DESTINATION examples/example2_old/)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/package/female_s1.wav DESTINATION examples/example2_old/)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/package/LICENCE.txt DESTINATION examples/example2_old/)
    
ENDIF()
