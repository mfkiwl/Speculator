//
//  pv.h
//  
//
//
//

#ifndef ____pv__
#define ____pv__

#define bool int
#define true 1
#define false 0

#include <stdio.h>
#include <math.h>
#include "internal.h"

typedef void *Phase_vocoder; // a phase vocoder
typedef struct {
    void *(*malloc)(size_t); // malloc is used to allocate memory. If you 
        // have a real-time system and want to avoid the standard library
        // malloc() which may have priority inversion problems due to locks,
        // you can supply your own lock-free implementation
    void (*free)(void *); // if you provide a custom malloc, you should 
        // provide a matching custom free()
    long blocksize; // the size of audio blocks which are the units of 
        // audio that are provided to the phase vocoder and the units
        // that are produced by it
    int fftsize; // the number of samples in each FFT. Should be power of 2.
    int syn_hopsize; // the hopsize used in reconstructing the output
    float ratio; // the time-stretch ratio.
    float pre_ratio; // previous ratio is used to calculate
        // the previous input_length
    float *ana_win; // the window function used on input (analysis)
    float *syn_win; // the window function used on output (synthesis)
    long input_eff_pos; // input effective position is the sample number
        // of the input that corresponds to the current output
    bool first_time;  // a sign to show if it's the FIRST frame
        // being processed
    bool initialized;  // a sign to show if the memory was allocated
    float *input_buffer; // used to buffer input samples
    float *output_buffer; // used to buffer output samples
    float *output; // the block of samples delivered as output
    float *input_frame_head; // pointer to the beginning of the frame 
        // to be read into the input buffer
    float *input_rear; // pointer to the end of the input data
    float *io_pos; // pointer to the position where we should get input
        // or give output in the output buffer;
    float *count_pointer; // move in output buffer to see if we need to 
        // get input or give output, count_pointer
        // is updated in pv_get_output();
    float *ana_frame; // analysis frame
    float *syn_frame; // synthesis frame
    float *realpart;  // real part of the output of FFT or IFFT
    float *imagpart;  // imaginary part of the output of FFT or IFFT
    float *mag; // magnitude for points in the frame being processed
    float *ana_phase; // phase for points in the analysis frame
                      // being processed
    float *syn_phase; // phase for points in the synthesis frame
                      // being processed
    float *pre_ana_phase; // recording last analysis phase for estimating
                          // the frequency
    float *pre_syn_phase; // recording last systhesis phase for rebuilting
                          // the new phase
    float *bin_freq; // bin frequency, used in phase unwrapping;
    float *phase_increment; // increment between actual phase increment value
                            // and the phase increment value got when the
                            // it's the nearest bin frequency. Used
                            // in phase unwrapping
    float *estimate_freq; // estimated frequency from phase unwrapping
    
    struct tag       // each element in the structure array
    {
        long ana_tag;
        long syn_tag;
    };
    struct tag *tag_buffer; // Circular array restoring the sample
                            // number of the middle of the frames
                            // (both for analysis and synthesis frames)
    struct tag *tag_buffer_head; // pointer to the beginning of the circular array
    struct tag *tag_buffer_rear; // pointer to the rear of the circular array
    long queue_length; // length of the circular queue
} PV;

// create a phase vocoder. Pass in function pointers to allocate
// and deallocate memory, or NULL to use default malloc()
// and free():
Phase_vocoder pv_create(void *(*malloc)(size_t), void (*free)(void *));

// call at the end of the program to free the memory
void pv_end(Phase_vocoder x);

// set output block size (call before computing first samples)
// must be a power of 2
// Blocksize should be set beforehand and can't be changed
// when program begins
void pv_set_blocksize(Phase_vocoder x, long n);

// set FFT size, n must be power of 2 (call before computing
// first samples)
// FFT size should be set beforehand and can't be changed
// when program begins
void pv_set_fftsize(Phase_vocoder x, int n);

// set ratio (call before pv_get_input_count())
void pv_set_ratio(Phase_vocoder x, float ratio);

// set synthesis hopsize, 'n' must be power of 2.
// It's better to restrict syn_hopsize can only be
// fftsize/2, fftsize/4, fftsize/8, fftsize/16.
// Synthesis hopsize should be set beforehand and can't be
// changed when program begins
void pv_set_syn_hopsize(Phase_vocoder x, int n);

// initialize: allocate space for window, set default window,
// allocate space for other parameters
// Note that the user won't have the default window and don't
// need to free it.
void pv_initialize(Phase_vocoder x);

// set analysis window (if it's not called, default is Hanning window)
// A copy of the window is made and managed by the Phase_vocoder, this
// copy will be freed by the phase vocoder( by calling pv_end() )
// this function can be called many times, but only the newest window set
// by the user is valid.
void pv_set_ana_window(Phase_vocoder x, float *window);

// set synthesis window, same requirement as "pv_set_ana_window"
void pv_set_syn_window(Phase_vocoder x, float *window);

// allocate a window and initialize it with the window_type function
// (see hann() and hamm() as example window_type functions)
// caller becomes the owner of the result, so the owner should free
// the window eventually.
// Note the output is after normalized to make sure the sum of w1*w2
// equals to 1 rather than other constants.
float *pv_window(Phase_vocoder x, float (*window_type)(double x));

// inquire how many samples needed to compute next output
long pv_get_input_count(Phase_vocoder x);

// get effective position of the next output sample 
// measured in samples of input. The user should send in
// the number of the next sample to be played
// if the return value is zero, the input sample_number must be invalid
float pv_get_effective_pos(Phase_vocoder x, long sample_number);

// send input samples - writes input to be stretched into
// the phase vocoder. Phase vocoder saves as much of the
// signal as needed so that input is always sequential.
// note the "size" should be the output of pv_get_input_count().
void pv_put_input(Phase_vocoder x, long size, float *samples);

// get output samples: returns a pointer to n samples, where
// n is the value from pv_set_block_size(). Pointer should
// not be freed by the user.
float *pv_get_output(Phase_vocoder x);

#endif /* defined(____pv__) */
