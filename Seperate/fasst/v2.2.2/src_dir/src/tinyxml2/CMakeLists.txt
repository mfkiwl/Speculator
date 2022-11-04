add_library(tinyxml2 STATIC tinyxml2.cpp tinyxml2.h)
set_target_properties(tinyxml2 PROPERTIES
        COMPILE_DEFINITONS "TINYXML2_EXPORT"
        VERSION "${GENERIC_LIB_VERSION}"
        SOVERSION "${GENERIC_LIB_SOVERSION}")
set_target_properties( tinyxml2 PROPERTIES OUTPUT_NAME tinyxml2 )

target_compile_definitions(tinyxml2 PUBLIC -D_CRT_SECURE_NO_WARNINGS)

if(DEFINED CMAKE_VERSION AND NOT "${CMAKE_VERSION}" VERSION_LESS "2.8.11")
    target_include_directories(tinyxml2 PUBLIC 
                          $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

    if(MSVC)
      target_compile_definitions(tinyxml2 PUBLIC -D_CRT_SECURE_NO_WARNINGS)
    endif(MSVC)
else()
    include_directories(${PROJECT_SOURCE_DIR})

    if(MSVC)
      add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    endif(MSVC)
endif()