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
include examples/talk/CMakeFiles/robot.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/talk/CMakeFiles/robot.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/talk/CMakeFiles/robot.dir/progress.make

# Include the compile flags for this target's objects.
include examples/talk/CMakeFiles/robot.dir/flags.make

examples/talk/CMakeFiles/robot.dir/talk.cpp.o: examples/talk/CMakeFiles/robot.dir/flags.make
examples/talk/CMakeFiles/robot.dir/talk.cpp.o: ../examples/talk/talk.cpp
examples/talk/CMakeFiles/robot.dir/talk.cpp.o: examples/talk/CMakeFiles/robot.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/talk/CMakeFiles/robot.dir/talk.cpp.o"
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/talk && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/talk/CMakeFiles/robot.dir/talk.cpp.o -MF CMakeFiles/robot.dir/talk.cpp.o.d -o CMakeFiles/robot.dir/talk.cpp.o -c /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/examples/talk/talk.cpp

examples/talk/CMakeFiles/robot.dir/talk.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/robot.dir/talk.cpp.i"
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/talk && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/examples/talk/talk.cpp > CMakeFiles/robot.dir/talk.cpp.i

examples/talk/CMakeFiles/robot.dir/talk.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/robot.dir/talk.cpp.s"
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/talk && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/examples/talk/talk.cpp -o CMakeFiles/robot.dir/talk.cpp.s

# Object files for target robot
robot_OBJECTS = \
"CMakeFiles/robot.dir/talk.cpp.o"

# External object files for target robot
robot_EXTERNAL_OBJECTS =

bin/robot: examples/talk/CMakeFiles/robot.dir/talk.cpp.o
bin/robot: examples/talk/CMakeFiles/robot.dir/build.make
bin/robot: examples/libcommon.a
bin/robot: examples/libcommon-sdl.a
bin/robot: libwhisper.so
bin/robot: examples/talk/libInferLLM.a
bin/robot: /usr/lib/libSDL2main.a
bin/robot: /usr/lib/libSDL2-2.0.so.0.2600.5
bin/robot: examples/talk/CMakeFiles/robot.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/robot"
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/talk && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/robot.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/talk/CMakeFiles/robot.dir/build: bin/robot
.PHONY : examples/talk/CMakeFiles/robot.dir/build

examples/talk/CMakeFiles/robot.dir/clean:
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/talk && $(CMAKE_COMMAND) -P CMakeFiles/robot.dir/cmake_clean.cmake
.PHONY : examples/talk/CMakeFiles/robot.dir/clean

examples/talk/CMakeFiles/robot.dir/depend:
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/examples/talk /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/talk /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/examples/talk/CMakeFiles/robot.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/talk/CMakeFiles/robot.dir/depend

