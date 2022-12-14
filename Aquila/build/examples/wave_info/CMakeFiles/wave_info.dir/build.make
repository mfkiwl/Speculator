# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/quake/Projects/Current/SoundWave/c++/Aquila

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/quake/Projects/Current/SoundWave/c++/Aquila/build

# Include any dependencies generated for this target.
include examples/wave_info/CMakeFiles/wave_info.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/wave_info/CMakeFiles/wave_info.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/wave_info/CMakeFiles/wave_info.dir/progress.make

# Include the compile flags for this target's objects.
include examples/wave_info/CMakeFiles/wave_info.dir/flags.make

examples/wave_info/CMakeFiles/wave_info.dir/wave_info.cpp.o: examples/wave_info/CMakeFiles/wave_info.dir/flags.make
examples/wave_info/CMakeFiles/wave_info.dir/wave_info.cpp.o: /home/quake/Projects/Current/SoundWave/c++/Aquila/examples/wave_info/wave_info.cpp
examples/wave_info/CMakeFiles/wave_info.dir/wave_info.cpp.o: examples/wave_info/CMakeFiles/wave_info.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/quake/Projects/Current/SoundWave/c++/Aquila/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/wave_info/CMakeFiles/wave_info.dir/wave_info.cpp.o"
	cd /home/quake/Projects/Current/SoundWave/c++/Aquila/build/examples/wave_info && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/wave_info/CMakeFiles/wave_info.dir/wave_info.cpp.o -MF CMakeFiles/wave_info.dir/wave_info.cpp.o.d -o CMakeFiles/wave_info.dir/wave_info.cpp.o -c /home/quake/Projects/Current/SoundWave/c++/Aquila/examples/wave_info/wave_info.cpp

examples/wave_info/CMakeFiles/wave_info.dir/wave_info.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wave_info.dir/wave_info.cpp.i"
	cd /home/quake/Projects/Current/SoundWave/c++/Aquila/build/examples/wave_info && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/quake/Projects/Current/SoundWave/c++/Aquila/examples/wave_info/wave_info.cpp > CMakeFiles/wave_info.dir/wave_info.cpp.i

examples/wave_info/CMakeFiles/wave_info.dir/wave_info.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wave_info.dir/wave_info.cpp.s"
	cd /home/quake/Projects/Current/SoundWave/c++/Aquila/build/examples/wave_info && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/quake/Projects/Current/SoundWave/c++/Aquila/examples/wave_info/wave_info.cpp -o CMakeFiles/wave_info.dir/wave_info.cpp.s

# Object files for target wave_info
wave_info_OBJECTS = \
"CMakeFiles/wave_info.dir/wave_info.cpp.o"

# External object files for target wave_info
wave_info_EXTERNAL_OBJECTS =

examples/wave_info/wave_info: examples/wave_info/CMakeFiles/wave_info.dir/wave_info.cpp.o
examples/wave_info/wave_info: examples/wave_info/CMakeFiles/wave_info.dir/build.make
examples/wave_info/wave_info: libAquila.a
examples/wave_info/wave_info: lib/libOoura_fft.a
examples/wave_info/wave_info: examples/wave_info/CMakeFiles/wave_info.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/quake/Projects/Current/SoundWave/c++/Aquila/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable wave_info"
	cd /home/quake/Projects/Current/SoundWave/c++/Aquila/build/examples/wave_info && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/wave_info.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/wave_info/CMakeFiles/wave_info.dir/build: examples/wave_info/wave_info
.PHONY : examples/wave_info/CMakeFiles/wave_info.dir/build

examples/wave_info/CMakeFiles/wave_info.dir/clean:
	cd /home/quake/Projects/Current/SoundWave/c++/Aquila/build/examples/wave_info && $(CMAKE_COMMAND) -P CMakeFiles/wave_info.dir/cmake_clean.cmake
.PHONY : examples/wave_info/CMakeFiles/wave_info.dir/clean

examples/wave_info/CMakeFiles/wave_info.dir/depend:
	cd /home/quake/Projects/Current/SoundWave/c++/Aquila/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/quake/Projects/Current/SoundWave/c++/Aquila /home/quake/Projects/Current/SoundWave/c++/Aquila/examples/wave_info /home/quake/Projects/Current/SoundWave/c++/Aquila/build /home/quake/Projects/Current/SoundWave/c++/Aquila/build/examples/wave_info /home/quake/Projects/Current/SoundWave/c++/Aquila/build/examples/wave_info/CMakeFiles/wave_info.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/wave_info/CMakeFiles/wave_info.dir/depend

