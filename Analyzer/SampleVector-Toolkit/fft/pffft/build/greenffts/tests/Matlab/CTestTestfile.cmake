# CMake generated Testfile for 
# Source directory: /home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab
# Build directory: /home/quake/Projects/audio-analysis/src/pffft/build/greenffts/tests/Matlab
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(convTestC "convTest")
set_tests_properties(convTestC PROPERTIES  _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/CMakeLists.txt;16;add_test;/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/CMakeLists.txt;0;")
add_test(conv2dTestC "conv2dTest")
set_tests_properties(conv2dTestC PROPERTIES  _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/CMakeLists.txt;17;add_test;/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/CMakeLists.txt;0;")
add_test(rfft2dTestMLC "rfft2dTestML")
set_tests_properties(rfft2dTestMLC PROPERTIES  _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/CMakeLists.txt;18;add_test;/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/CMakeLists.txt;0;")
add_test(convTestM "octave" "--no-history" "--quiet" "/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/convtest.m")
set_tests_properties(convTestM PROPERTIES  _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/CMakeLists.txt;20;add_test;/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/CMakeLists.txt;0;")
add_test(conv2dTestM "octave" "--no-history" "--quiet" "/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/conv2dtest.m")
set_tests_properties(conv2dTestM PROPERTIES  _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/CMakeLists.txt;25;add_test;/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/CMakeLists.txt;0;")
add_test(rfft2dTestMLM "octave" "--no-history" "--quiet" "/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/rfft2dTestML.m")
set_tests_properties(rfft2dTestMLM PROPERTIES  _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/CMakeLists.txt;30;add_test;/home/quake/Projects/audio-analysis/src/pffft/greenffts/tests/Matlab/CMakeLists.txt;0;")
