cmake_minimum_required(VERSION 2.6.2)

project( googletest-distribution )

enable_testing()

option(BUILD_GTEST "Builds the googletest subproject" OFF)

#Note that googlemock target already builds googletest
option(BUILD_GMOCK "Builds the googlemock subproject" ON)

# [EC/RLB] Work around due to issue : https://github.com/Microsoft/TestAdapterForGoogleTest/issues/119
if(MSVC)
    ADD_DEFINITIONS(/D _SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
endif()

if(BUILD_GMOCK)
  add_subdirectory( googlemock )
  SET_TARGET_PROPERTIES (gtest gtest_main PROPERTIES FOLDER GTEST)
  SET_TARGET_PROPERTIES (gmock gmock_main PROPERTIES FOLDER GTEST)
elseif(BUILD_GTEST)
  add_subdirectory( googletest )
  SET_TARGET_PROPERTIES (gtest gtest_main PROPERTIES FOLDER GTEST)
endif()
