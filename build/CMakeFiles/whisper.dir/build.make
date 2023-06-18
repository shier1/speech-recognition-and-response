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
include CMakeFiles/whisper.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/whisper.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/whisper.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/whisper.dir/flags.make

CMakeFiles/whisper.dir/ggml.c.o: CMakeFiles/whisper.dir/flags.make
CMakeFiles/whisper.dir/ggml.c.o: ../ggml.c
CMakeFiles/whisper.dir/ggml.c.o: CMakeFiles/whisper.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/whisper.dir/ggml.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/whisper.dir/ggml.c.o -MF CMakeFiles/whisper.dir/ggml.c.o.d -o CMakeFiles/whisper.dir/ggml.c.o -c /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/ggml.c

CMakeFiles/whisper.dir/ggml.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/whisper.dir/ggml.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/ggml.c > CMakeFiles/whisper.dir/ggml.c.i

CMakeFiles/whisper.dir/ggml.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/whisper.dir/ggml.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/ggml.c -o CMakeFiles/whisper.dir/ggml.c.s

CMakeFiles/whisper.dir/whisper.cpp.o: CMakeFiles/whisper.dir/flags.make
CMakeFiles/whisper.dir/whisper.cpp.o: ../whisper.cpp
CMakeFiles/whisper.dir/whisper.cpp.o: CMakeFiles/whisper.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/whisper.dir/whisper.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/whisper.dir/whisper.cpp.o -MF CMakeFiles/whisper.dir/whisper.cpp.o.d -o CMakeFiles/whisper.dir/whisper.cpp.o -c /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/whisper.cpp

CMakeFiles/whisper.dir/whisper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/whisper.dir/whisper.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/whisper.cpp > CMakeFiles/whisper.dir/whisper.cpp.i

CMakeFiles/whisper.dir/whisper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/whisper.dir/whisper.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/whisper.cpp -o CMakeFiles/whisper.dir/whisper.cpp.s

# Object files for target whisper
whisper_OBJECTS = \
"CMakeFiles/whisper.dir/ggml.c.o" \
"CMakeFiles/whisper.dir/whisper.cpp.o"

# External object files for target whisper
whisper_EXTERNAL_OBJECTS =

libwhisper.so: CMakeFiles/whisper.dir/ggml.c.o
libwhisper.so: CMakeFiles/whisper.dir/whisper.cpp.o
libwhisper.so: CMakeFiles/whisper.dir/build.make
libwhisper.so: CMakeFiles/whisper.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libwhisper.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/whisper.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/whisper.dir/build: libwhisper.so
.PHONY : CMakeFiles/whisper.dir/build

CMakeFiles/whisper.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/whisper.dir/cmake_clean.cmake
.PHONY : CMakeFiles/whisper.dir/clean

CMakeFiles/whisper.dir/depend:
	cd /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build /home/shier/curriculum_design/speech_singal_processing/design/speech_recognize/build/CMakeFiles/whisper.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/whisper.dir/depend

