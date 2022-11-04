CMAKE_MINIMUM_REQUIRED(VERSION 3.9.3)
PROJECT(FASST)

# Set FASST version
SET (FASST_VERSION 2.2.0)

# Cmake configuration : use solution folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

IF(MSVC)
    # add version for windows installer
    SET(CPACK_PACKAGE_VERSION ${FASST_VERSION})

    # add version for cmake install directory
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}_${FASST_VERSION}")
ENDIF()

IF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}_${FASST_VERSION}")

    # Parse necessary files for os x installer
    ADD_SUBDIRECTORY(packaging)
ENDIF()

# Compiler-specific C++11 activation. useful only for OS X ? To be verified
IF("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    IF(NOT (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7))
        message(FATAL_ERROR "${PROJECT_NAME} requires g++ 4.7 or greater.")
    ENDIF()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

ELSEIF("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") #eg if(os x)
    message(FATAL_ERROR "Your C++ compiler does not support C++11.")
ENDIF()

# Specify output directory for every platform
IF(MSVC)
    # CMAKE_RUNTIME_OUTPUT_DIRECTORY is the input directory used by NSIS for input dll that will be used for windows installer
    # => Installer dll are set to correspond to release binaries !
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Release)
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
ELSE()
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
ENDIF()

# Build Release by default (it needs to be manually selected on MSCV)
IF(NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "Release"
        CACHE STRING "[Debug|Release|RelWithDebInfo|MixSizeRel]"
        FORCE)
ENDIF()

# Deal with custom arguments :
# EIGEN_DONT_PARALLELIZE : Disable Eigen parallelization (apparently slow down the code)
IF(EIGEN_DONT_PARALLELIZE)
    ADD_DEFINITIONS(-DEIGEN_DONT_PARALLELIZE)
    MESSAGE(STATUS "Eigen parallelization disabled")
ELSE(EIGEN_DONT_PARALLELIZE)
    MESSAGE(STATUS "Eigen parallelization enabled")
ENDIF(EIGEN_DONT_PARALLELIZE)

IF(EIGEN_NO_DEBUG)
    ADD_DEFINITIONS(-DEIGEN_NO_DEBUG)
    MESSAGE(STATUS "Eigen NO DEBUG enabled")
ENDIF()
#ADD_DEFINITIONS(-DEIGEN_MAX_ALIGN_BYTES=16)


# Make sure Microsoft Visual Studio knows about M_PI
IF(MSVC)
    ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
ENDIF()

# Try to find OpenMP
FIND_PACKAGE(OpenMP)
IF(OPENMP_CXX_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

#Define library suffixes and prefixes depending of OS
IF(WIN32)
    SET(CMAKE_FIND_LIBRARY_PREFIXES "")
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
ELSE(WIN32)
    SET(CMAKE_FIND_LIBRARY_PREFIXES "lib")
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a")
ENDIF(WIN32)


# Find libsndfile
#----------------------------------
IF(UNIX)
    SET(SNDFILE_LIB sndfile)
ELSEIF(WIN32)
    FIND_PATH(LIBSNDFILE_INCLUDE_DIR sndfile.h "C:/Program Files (x86)/Mega-Nerd/libsndfile/include" "C:/Program Files/Mega-Nerd/libsndfile/include")
    FIND_LIBRARY(SNDFILE_LIB libsndfile-1.lib "C:/Program Files (x86)/Mega-Nerd/libsndfile/lib" "C:/Program Files/Mega-Nerd/libsndfile/lib" PATH)
    FIND_LIBRARY(SNDFILE_DLL libsndfile-1.dll "C:/Program Files (x86)/Mega-Nerd/libsndfile/bin" "C:/Program Files/Mega-Nerd/libsndfile/bin" PATH)
    #add include directory
    INCLUDE_DIRECTORIES(${LIBSNDFILE_INCLUDE_DIR})

    IF(LIBSNDFILE_INCLUDE_DIR AND SNDFILE_LIB AND SNDFILE_DLL)
        MESSAGE(STATUS "sndFile: FOUND")
    ELSE()
        MESSAGE(FATAL_ERROR "Could not find sndFile")
    ENDIF(LIBSNDFILE_INCLUDE_DIR AND SNDFILE_LIB AND SNDFILE_DLL)
ENDIF()

# Find eigen
IF(UNIX)
    SET(EIGEN_PATH /usr/local/include/eigen3 /usr/include/eigen3)
ELSEIF(WIN32)
    SET(EIGEN_PATH "C:/Program\ Files\ (x86)/Eigen3/include/eigen3")
ENDIF()
INCLUDE_DIRECTORIES(${EIGEN_PATH})

#Try to find FFTW if requested
if(EIGEN_FFTW_DEFAULT)
    INCLUDE(./FindFFTW.cmake)
    ADD_DEFINITIONS(-DEIGEN_FFTW_DEFAULT)
    INCLUDE_DIRECTORIES(${FFTW_INCLUDES})
ENDIF()

# inlude for tinyxml2
#Add include directory (tinyxml2)
INCLUDE_DIRECTORIES("./src/tinyxml2/")

# Try to find Doxygen
FIND_PACKAGE(Doxygen)
IF(DOXYGEN_FOUND)
    CONFIGURE_FILE(doc/Doxyfile.in doc/Doxyfile @ONLY)
    ADD_CUSTOM_TARGET(doc ALL
        ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/doc/Doxyfile
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc
        VERBATIM)
        SET_TARGET_PROPERTIES (doc PROPERTIES FOLDER Doxygen)
ENDIF(DOXYGEN_FOUND)

# If the TEST variable is set to true, activate tests and build test framework

IF(TEST)
    ENABLE_TESTING()


    # Build googletest framework
    IF(MSVC)
        ADD_DEFINITIONS(-D_VARIADIC_MAX=10)
    ENDIF()
    ADD_SUBDIRECTORY(gtest-1.8.0)
    INCLUDE_DIRECTORIES(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
    IF(MSVC)
        # Be sure to correctly link against googletest framework
        SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} /MTd /D_DEBUG")
        #RL : correctly set debug information format for test projects
        SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")
        SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} /MT /D_RELEASE")
    ELSE(NOT MSVC)
        SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -D_DEBUG")
        SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_RELEASE")
    ENDIF()
ENDIF()


IF(MSVC)

    # Add strict warnings
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /WX")

        # disable specific warnings for debug
        #4996 : 'std::copy::_Unchecked_iterators::_Deprecate': Call to 'std::copy' with parameters that may be unsafe - this call relies on the caller to check that the passed values are correct.
        #4996 (release)	'std::tr1': warning STL4002: The non-Standard std::tr1 namespace and TR1-only machinery are deprecated and will be REMOVED. You can define _SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING to acknowledge that you have received this warning.
        #4503 : for eigen, "decorated name length exceeded, name was truncated"
        #4805 : for fftw mixture of bool and int type - to be checked for proper correction
        FOREACH(warningDebug  4996 4503 4805)
             SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /wd${warningDebug}")
        ENDFOREACH(warningDebug)

        # disable specific warnings for Release
        FOREACH(warningRelease 4996 4503 4805)
            SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /wd${warningRelease}")
        ENDFOREACH(warningRelease)

ELSE()
    # Add O2 optimization(default value for MSCV)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

    # Add strict warnings (warning considered as error)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

    # clang compiler doesn't know #pragma omp (disable unknown pragma warning)
    IF(CMAKE_CXX_COMPILER_ID MATCHES "clang")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unknown-pragmas")
    ENDIF()

    # if gcc version > 4.7, another warning is triggered by Eigen
    EXECUTE_PROCESS(COMMAND ${CMAKE_C_COMPILER} -dumpversion
      OUTPUT_VARIABLE GCC_VERSION)
    IF(GCC_VERSION VERSION_GREATER 4.7)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=aggressive-loop-optimizations")
    ENDIF()

    #Disable specific warning for GCC:
    #Wno-int-in-bool-context : for eigen 3.3.4, must be solved in future release
    IF(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-in-bool-context")
    ENDIF()
ENDIF()

# Build fasst source tree
ADD_SUBDIRECTORY(src)

# Configure scripts
ADD_SUBDIRECTORY(scripts)

# Configure examples
ADD_SUBDIRECTORY(examples)

# Build a Windows installer
IF(MSVC)
    # Use the modified NSIS template
    
    SET(CPACK_PACKAGE_VENDOR "INRIA")
    set(CPACK_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/packaging/win/")

    SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

    string (REPLACE ";" "\n" CPACK_NSIS_EXTRA_PREINSTALL_COMMANDS "${CPACK_NSIS_EXTRA_PREINSTALL_COMMANDS}")
    #for CPACK windows installer :
    # move examples folder to $ExInstDir (selected by user into the gui)
    # don't change the name of the subfolder "fasst_examples"
    SET(CPACK_NSIS_EXTRA_INSTALL_COMMANDS
    "  CopyFiles '$INSTDIR\\\\examples\\\\example1\\\\*.*' '$ExInstDir\\\\${PROJECT_NAME}_${FASST_VERSION}_examples\\\\example1'"
    "  CopyFiles '$INSTDIR\\\\examples\\\\example2\\\\*.*' '$ExInstDir\\\\${PROJECT_NAME}_${FASST_VERSION}_examples\\\\example2'"
    "  CopyFiles '$INSTDIR\\\\examples\\\\example3\\\\*.*' '$ExInstDir\\\\${PROJECT_NAME}_${FASST_VERSION}_examples\\\\example3'"
    "  CopyFiles '$INSTDIR\\\\examples\\\\tools\\\\*.*' '$ExInstDir\\\\${PROJECT_NAME}_${FASST_VERSION}_examples\\\\tools'"
    "  CopyFiles '$INSTDIR\\\\examples\\\\QuickStarter.txt' '$ExInstDir\\\\${PROJECT_NAME}_${FASST_VERSION}_examples\\\\'"
    "  CopyFiles '$INSTDIR\\\\examples\\\\install_peass.bat' '$ExInstDir\\\\${PROJECT_NAME}_${FASST_VERSION}_examples\\\\'"
    "  RMDir /r '$INSTDIR\\\\examples'")
    # replace ; by \n in CPACK_NSIS_EXTRA_INSTALL_COMMANDS (multiple commands)
    string (REPLACE ";" "\n" CPACK_NSIS_EXTRA_INSTALL_COMMANDS "${CPACK_NSIS_EXTRA_INSTALL_COMMANDS}")

    #for CPACK windows uninstaller : delete fasst_example folder in $ExInstDir
    SET(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS
    "  RMDir /r '$ExInstDir\\\\fasst_examples'")
    # replace ; by \n in CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS (multiple commands)
    string (REPLACE ";" "\n" CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "${CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS}")

    # Add visual studio dll
    INCLUDE(InstallRequiredSystemLibraries)

    # Add openmp dll
    IF(OPENMP_FOUND)
        IF(CMAKE_CL_64)
            SET(MSVCSUBDIR "x64")
        ELSE()
            SET(MSVCSUBDIR "x86")
        ENDIF()
        IF(MSVC14 AND ${MSVC_VERSION} GREATER_EQUAL 1911) # Test for MSVC 2015 (2017 version)
        # For MSVC15 (2017 version), MSVC15 variable is not defined, but MSVC14 variable is defined.
        # Have a look to : https://gitlab.kitware.com/cmake/cmake/issues/16735
                INSTALL(FILES
                "${MSVC_REDIST_DIR}/${MSVCSUBDIR}/Microsoft.VC141.OPENMP/vcomp140.dll"
                DESTINATION bin)
        ELSEIF(MSVC14)
                INSTALL(FILES
                "${MSVC_REDIST_DIR}/${MSVCSUBDIR}/Microsoft.VC140.OPENMP/vcomp140.dll"
                DESTINATION bin)
        ELSEIF(MSVC12)
                INSTALL(FILES
                "${MSVC_REDIST_DIR}/${MSVCSUBDIR}/Microsoft.VC120.OPENMP/vcomp120.dll"
                DESTINATION bin)
        ELSEIF(MSVC11)
                INSTALL(FILES
                "${MSVC_REDIST_DIR}/${MSVCSUBDIR}/Microsoft.VC110.OPENMP/vcomp110.dll"
                DESTINATION bin)
        ENDIF()
    ENDIF()

    # Force the install path to be C:\Program Files
    SET(CPACK_NSIS_INSTALL_ROOT "C:\\\\Program Files")

    SET(CPACK_PACKAGE_INSTALL_DIRECTORY "${PROJECT_NAME}_${FASST_VERSION}")

    # Create an installer
    INCLUDE(CPack)
ENDIF()
