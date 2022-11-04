# JUCE library targets for HybridReverb2

set(JUCE_ROOT_DIR "${CMAKE_SOURCE_DIR}/Thirdparty/JUCE")
include(JUCE)

add_juce_module(juce_core)
add_juce_module(juce_events)
add_juce_module(juce_data_structures)
add_juce_module(juce_audio_basics)
add_juce_module(juce_audio_devices)
add_juce_module(juce_audio_formats)
add_juce_module(juce_audio_processors)
add_juce_module(juce_audio_utils)
add_juce_module(juce_graphics)
add_juce_module(juce_gui_basics)
add_juce_module(juce_gui_extra)
target_link_libraries(juce_events PUBLIC juce_core)
target_link_libraries(juce_data_structures PUBLIC juce_events)
target_link_libraries(juce_audio_basics PUBLIC juce_core)
target_link_libraries(juce_audio_devices PUBLIC juce_audio_basics juce_events)
target_link_libraries(juce_audio_formats PUBLIC juce_audio_basics)
target_link_libraries(juce_audio_processors PUBLIC juce_gui_extra juce_audio_basics)
target_link_libraries(juce_audio_utils PUBLIC juce_gui_extra juce_audio_processors juce_audio_formats juce_audio_devices)
target_link_libraries(juce_gui_basics PUBLIC juce_graphics juce_data_structures)
target_link_libraries(juce_gui_extra PUBLIC juce_gui_basics)

# define the name of the Jack client
target_compile_definitions(juce_audio_devices PRIVATE "JUCE_JACK_CLIENT_NAME=\"HybridReverb2\"")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # need this circular link dependency on Windows
  target_link_libraries(juce_events PUBLIC juce_gui_extra)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
else()
  # need this circular link dependency on X11
  target_link_libraries(juce_gui_basics PUBLIC juce_gui_extra)
endif()

find_package(Threads REQUIRED)
target_link_libraries(juce_core PRIVATE ${CMAKE_THREAD_LIBS_INIT})

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # Windows
  target_link_libraries(juce_core PRIVATE winmm wininet ws2_32 shlwapi version)
  target_link_libraries(juce_gui_basics PRIVATE imm32)
  #
  set(HybridReverb2_ASIO_SDK "${PROJECT_SOURCE_DIR}/Thirdparty/ASIOSDK2.3" CACHE STRING "ASIO SDK location")
  if(NOT EXISTS "${HybridReverb2_ASIO_SDK}/common/iasiodrv.h")
    message(FATAL_ERROR "ASIO SDK not found in directory ${HybridReverb2_ASIO_SDK}")
  endif()
  target_include_directories(juce_audio_devices PRIVATE "${HybridReverb2_ASIO_SDK}/common")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  # Mac
  find_library(FOUNDATION_LIBRARY "Foundation")
  find_library(COCOA_LIBRARY "Cocoa")
  find_library(CORE_SERVICES_LIBRARY "CoreServices")
  find_library(APPLICATION_SERVICES_LIBRARY "ApplicationServices")
  find_library(CORE_AUDIO_LIBRARY "CoreAudio")
  find_library(CORE_MIDI_LIBRARY "CoreMIDI")
  find_library(AUDIO_TOOLBOX_LIBRARY "AudioToolbox")
  find_library(ACCELERATE_LIBRARY "Accelerate")
  find_library(CORE_IMAGE_LIBRARY "CoreImage")
  find_library(IOKIT_LIBRARY "IOKit")
  target_link_libraries(juce_core PRIVATE "${FOUNDATION_LIBRARY}")
  target_link_libraries(juce_core PRIVATE "${COCOA_LIBRARY}")
  target_link_libraries(juce_core PRIVATE "${CORE_SERVICES_LIBRARY}")
  target_link_libraries(juce_core PRIVATE "${APPLICATION_SERVICES_LIBRARY}")
  target_link_libraries(juce_audio_basics PRIVATE "${ACCELERATE_LIBRARY}")
  target_link_libraries(juce_audio_devices PRIVATE "${CORE_AUDIO_LIBRARY}")
  target_link_libraries(juce_audio_devices PRIVATE "${CORE_MIDI_LIBRARY}")
  target_link_libraries(juce_audio_formats PRIVATE "${AUDIO_TOOLBOX_LIBRARY}")
  target_link_libraries(juce_graphics PRIVATE "${CORE_IMAGE_LIBRARY}")
  target_link_libraries(juce_gui_basics PRIVATE "${IOKIT_LIBRARY}")
else()
  # Linux and others
  include(FindPkgConfig)
  #
  target_link_libraries(juce_core PRIVATE "dl")
  #
  find_package(ALSA REQUIRED)
  target_link_libraries(juce_audio_devices PRIVATE ${ALSA_LIBRARIES})
  target_include_directories(juce_audio_devices PRIVATE ${ALSA_INCLUDE_DIRS})
  #
  pkg_check_modules(JACK "jack" REQUIRED)
  target_link_libraries(juce_audio_devices PRIVATE ${JACK_LIBRARIES})
  target_include_directories(juce_audio_devices PRIVATE ${JACK_INCLUDE_DIRS})
  link_directories(${JACK_LIBRARY_DIRS})
  #
  find_package(Freetype REQUIRED)
  target_link_libraries(juce_graphics PRIVATE ${FREETYPE_LIBRARIES})
  target_include_directories(juce_graphics PRIVATE ${FREETYPE_INCLUDE_DIRS})
  #
  find_package(X11 REQUIRED)
  target_link_libraries(juce_gui_basics PRIVATE ${X11_LIBRARIES})
  target_include_directories(juce_gui_basics PRIVATE ${X11_INCLUDE_DIRS})
  #
  if(FALSE)
    pkg_check_modules(GTK "gtk+-3.0" REQUIRED)
    target_link_libraries(juce_gui_extra PRIVATE ${GTK_LIBRARIES})
    target_include_directories(juce_gui_extra PRIVATE ${GTK_INCLUDE_DIRS})
    link_directories(${GTK_LIBRARY_DIRS})
  endif()
  # Curl (headers only)
  find_path(CURL_INCLUDE_DIR "curl/curl.h")
  if(NOT CURL_INCLUDE_DIR)
    message(FATAL_ERROR "Cannot find cURL headers")
  endif()
  message(STATUS "Found cURL headers: ${CURL_INCLUDE_DIR}")
  target_include_directories(juce_core PRIVATE "${CURL_INCLUDE_DIR}")
endif()

if(HybridReverb2_VST2)
  set(VST2_SOURCES
    "${PROJECT_SOURCE_DIR}/JuceLibraryCode/include_juce_audio_plugin_client_utils.cpp"
    "${PROJECT_SOURCE_DIR}/JuceLibraryCode/include_juce_audio_plugin_client_VST2.cpp")
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    list(APPEND VST2_SOURCES "${PROJECT_SOURCE_DIR}/JuceLibraryCode/include_juce_audio_plugin_client_VST_utils.mm")
  endif()
  add_juce_module(juce_audio_plugin_client_VST2 ${VST2_SOURCES})
  target_link_libraries(juce_audio_plugin_client_VST2 PUBLIC juce_gui_basics juce_audio_basics juce_audio_processors)
endif()

if(HybridReverb2_LV2)
  set(LV2_SOURCES
    "${PROJECT_SOURCE_DIR}/JuceLibraryCode/include_juce_audio_plugin_client_utils.cpp"
    "${PROJECT_SOURCE_DIR}/JuceLibraryCode/include_juce_audio_plugin_client_LV2.cpp")
  add_juce_module(juce_audio_plugin_client_LV2 ${LV2_SOURCES})
  target_link_libraries(juce_audio_plugin_client_LV2 PUBLIC juce_gui_basics juce_audio_basics juce_audio_processors)

  target_compile_definitions(juce_audio_plugin_client_LV2
    PUBLIC "JucePlugin_Build_LV2=1"
    PUBLIC "JucePlugin_Build_LV2_FileCreator=1"
    PUBLIC "JucePlugin_LV2URI=\"${HybridReverb2_URI}\"")
endif()

if(HybridReverb2_Standalone)
  set(Standalone_SOURCES
    "${PROJECT_SOURCE_DIR}/JuceLibraryCode/include_juce_audio_plugin_client_utils.cpp")
  if(NOT HybridReverb2_StandaloneCustom)
    list(APPEND Standalone_SOURCES
      "${PROJECT_SOURCE_DIR}/JuceLibraryCode/include_juce_audio_plugin_client_Standalone.cpp")
  endif()
  add_juce_module(juce_audio_plugin_client_Standalone ${Standalone_SOURCES})
  target_link_libraries(juce_audio_plugin_client_Standalone PUBLIC juce_gui_basics juce_audio_basics juce_audio_processors)
  if(HybridReverb2_NonSessionManager)
    target_compile_definitions(juce_audio_plugin_client_Standalone PRIVATE "WITH_NSM_SESSION_SUPPORT")
    target_include_directories(juce_audio_plugin_client_Standalone PRIVATE "${PROJECT_SOURCE_DIR}/Thirdparty/nonlib")
    target_include_directories(juce_audio_plugin_client_Standalone PRIVATE ${LIBLO_INCLUDE_DIRS})
    target_link_libraries(juce_audio_plugin_client_Standalone PRIVATE ${LIBLO_LIBRARIES})
    link_directories(${LIBLO_LIBRARY_DIRS})
  endif()
endif()
