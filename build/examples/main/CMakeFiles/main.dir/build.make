# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

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
CMAKE_SOURCE_DIR = /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build

# Include any dependencies generated for this target.
include examples/main/CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/main/CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/main/CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include examples/main/CMakeFiles/main.dir/flags.make

examples/main/CMakeFiles/main.dir/main.cpp.o: examples/main/CMakeFiles/main.dir/flags.make
examples/main/CMakeFiles/main.dir/main.cpp.o: ../examples/main/main.cpp
examples/main/CMakeFiles/main.dir/main.cpp.o: examples/main/CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/main/CMakeFiles/main.dir/main.cpp.o"
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/main && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/main/CMakeFiles/main.dir/main.cpp.o -MF CMakeFiles/main.dir/main.cpp.o.d -o CMakeFiles/main.dir/main.cpp.o -c /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/examples/main/main.cpp

examples/main/CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/main && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/examples/main/main.cpp > CMakeFiles/main.dir/main.cpp.i

examples/main/CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/main && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/examples/main/main.cpp -o CMakeFiles/main.dir/main.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

bin/main: examples/main/CMakeFiles/main.dir/main.cpp.o
bin/main: examples/main/CMakeFiles/main.dir/build.make
bin/main: examples/libcommon.a
bin/main: libwhisper.so
bin/main: examples/main/CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/main"
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/main && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/main/CMakeFiles/main.dir/build: bin/main
.PHONY : examples/main/CMakeFiles/main.dir/build

examples/main/CMakeFiles/main.dir/clean:
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/main && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : examples/main/CMakeFiles/main.dir/clean

examples/main/CMakeFiles/main.dir/depend:
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/examples/main /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/main /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/main/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/main/CMakeFiles/main.dir/depend

