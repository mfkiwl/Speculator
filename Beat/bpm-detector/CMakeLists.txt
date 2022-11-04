project(bpm-detector LANGUAGES C)

add_executable(bpm-detector bpm_detector.c kiss_fft/kiss_fft.c kiss_fft/kiss_fftr.c)
target_link_libraries(bpm-detector -lm)