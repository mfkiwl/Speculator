#Real-time Phase Vocoder Component

This is the real-time phase vocoder doing time-stretching in Human Computer Music Performance. Be sure to include libsndfile before using it.

“pv.h” is the head file that contains interfaces designation for phase vocoder.

“pv.c” is the file that contains the interfaces implementation.
    
“internal.h” and “internal.c” contains functions that used for internal computation for phase vocoder. It includes window function “Hann”, “Hamm”, and “FFTshift”.
    
“fftext.h/.c”, “fftlib.h/.c”, “matlib.h/.c” are files in FFT Library.