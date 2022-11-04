//
//  user.c
//  
//
//
//  These code are how the user use the phase vocoder
//  "FOR TEST" sign notes that libsndfile is used here to test the program

#include <stdio.h>
#include <stdlib.h>
#include "pv.h"

#include <sndfile.h>       /* FOR TEST */
#include <ncurses.h>      // used for mouse and keyboard control

int main()
{
    /* FOR TEST BEGIN */
    SNDFILE	*input_file;
    SF_INFO	input_info;
    input_info.format = 0;  // read only
    if (! (input_file = sf_open ("origin.wav", SFM_READ, &input_info)))
    {
        printf ("Error : Not able to open input file.\n") ;
        return 1 ;
    };
    int samplerate = input_info.samplerate;
    int channels = input_info.channels;
    int format = input_info.format;
    /* FOR TEST END */
    
    Phase_vocoder pv;
    long blocksize = 1000000;
    int fftsize = 2048;
    pv = pv_create(malloc, free);
    pv_set_blocksize(pv, blocksize);
    pv_set_fftsize(pv, fftsize);
    float *anawindow, *synwindow;
    anawindow = pv_window(pv, hann);
    synwindow = pv_window(pv, hann);
    pv_set_ana_window(pv, anawindow);
    pv_set_syn_window(pv, synwindow);
    
    /* from here, user can call follow functions many times */
    
    // ratio may be modified from the mouse or keyboard;
    // ratio > 1 means speed up, ratio < 1 means slow down
    float ratio = 0.5;
    pv_set_ratio(pv, ratio);
    
    long input_count;
    input_count = pv_get_input_count(pv);
    
    if (input_count)
    {
        float samples[input_count];     // need to be initialized by the sample points to be processed this time
        
        /* FOR TEST BEGIN */
        long input_num;
        input_num = sf_read_float(input_file, samples, input_count) ;
        /* FOR TEST END */
        
        pv_put_input(pv, input_count, samples);
    }
    
    /* FOR TEST BEGIN */
    SNDFILE *output_file;
    SF_INFO output_info;
    output_info.samplerate = samplerate;
    output_info.channels = channels;
    output_info.format = format;
    if (! (output_file = sf_open ("output.wav", SFM_WRITE, &output_info)))
    {
        printf ("Error : Not able to open output file.\n") ;
        return 1 ;
    };
    /* FOR TEST END */
    
    float *output;
    output = pv_get_output(pv);
    
    /* FOR TEST BEGIN */
    long output_num;
    output_num = sf_write_float(output_file, output, blocksize) ;
    /* FOR TEST END */
    
    //memcpy(mybuffer, output, blocksize * sizeof(float));    // "mybuffer" will be read by the next component
    
    /* the user call the above many times */
    
    free(anawindow);
    free(synwindow);
    pv_end(pv);
    
    /* FOR TEST BEGIN */
    int input_index;
    input_index = sf_close(input_file);
    int output_index;
    output_index = sf_close(output_file);
    /* FOR TEST END */
    
    return 0;
}
