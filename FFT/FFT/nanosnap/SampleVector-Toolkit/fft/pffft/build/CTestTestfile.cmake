# CMake generated Testfile for 
# Source directory: /home/quake/Projects/audio-analysis/src/pffft
# Build directory: /home/quake/Projects/audio-analysis/src/pffft/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_fft_factors "/home/quake/Projects/audio-analysis/src/pffft/build/test_fft_factors")
set_tests_properties(test_fft_factors PROPERTIES  WORKING_DIRECTORY "/home/quake/Projects/audio-analysis/src/pffft/build" _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;607;add_test;/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;0;")
add_test(test_fftpack_float "/home/quake/Projects/audio-analysis/src/pffft/build/test_fftpack_float")
set_tests_properties(test_fftpack_float PROPERTIES  WORKING_DIRECTORY "/home/quake/Projects/audio-analysis/src/pffft/build" _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;613;add_test;/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;0;")
add_test(test_fftpack_double "/home/quake/Projects/audio-analysis/src/pffft/build/test_fftpack_double")
set_tests_properties(test_fftpack_double PROPERTIES  WORKING_DIRECTORY "/home/quake/Projects/audio-analysis/src/pffft/build" _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;618;add_test;/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;0;")
add_test(bench_pffft_pow2 "/home/quake/Projects/audio-analysis/src/pffft/build/bench_pffft_float" "--max-len" "128" "--quick")
set_tests_properties(bench_pffft_pow2 PROPERTIES  WORKING_DIRECTORY "/home/quake/Projects/audio-analysis/src/pffft/build" _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;627;add_test;/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;0;")
add_test(bench_pffft_non2 "/home/quake/Projects/audio-analysis/src/pffft/build/bench_pffft_float" "--non-pow2" "--max-len" "192" "--quick")
set_tests_properties(bench_pffft_non2 PROPERTIES  WORKING_DIRECTORY "/home/quake/Projects/audio-analysis/src/pffft/build" _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;632;add_test;/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;0;")
add_test(test_pfconv_lens_symetric "/home/quake/Projects/audio-analysis/src/pffft/build/test_pffastconv" "--no-bench" "--quick" "--sym")
set_tests_properties(test_pfconv_lens_symetric PROPERTIES  WORKING_DIRECTORY "/home/quake/Projects/audio-analysis/src/pffft/build" _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;642;add_test;/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;0;")
add_test(test_pfconv_lens_non_sym "/home/quake/Projects/audio-analysis/src/pffft/build/test_pffastconv" "--no-bench" "--quick")
set_tests_properties(test_pfconv_lens_non_sym PROPERTIES  WORKING_DIRECTORY "/home/quake/Projects/audio-analysis/src/pffft/build" _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;647;add_test;/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;0;")
add_test(bench_pfconv_symetric "/home/quake/Projects/audio-analysis/src/pffft/build/test_pffastconv" "--no-len" "--quick" "--sym")
set_tests_properties(bench_pfconv_symetric PROPERTIES  WORKING_DIRECTORY "/home/quake/Projects/audio-analysis/src/pffft/build" _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;652;add_test;/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;0;")
add_test(bench_pfconv_non_sym "/home/quake/Projects/audio-analysis/src/pffft/build/test_pffastconv" "--no-len" "--quick")
set_tests_properties(bench_pfconv_non_sym PROPERTIES  WORKING_DIRECTORY "/home/quake/Projects/audio-analysis/src/pffft/build" _BACKTRACE_TRIPLES "/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;657;add_test;/home/quake/Projects/audio-analysis/src/pffft/CMakeLists.txt;0;")
subdirs("greenffts")
subdirs("kissfft")
subdirs("pocketfft")
subdirs("examples")
