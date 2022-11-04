#define  _CRT_SECURE_NO_WARNINGS
#include "timing.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "hsfft.h"
#include <vector>
#include "resample.h"
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// <RESAMPLING BASED ON FFT>
// https://www.dsprelated.com/showcode/54.php

void FFTResample(float *input, float *output, size_t input_size, size_t output_size) {
    fft_t *samples = (fft_t *) calloc(MAX(input_size, output_size), sizeof(fft_t));
    if (samples == NULL) {
        return;
    }
    fft_real_object fftPlan = fft_real_init(input_size, 1);
    fft_r2c_exec(fftPlan, input, samples);
    free_real_fft(fftPlan);
    if (output_size < input_size) {
        // remove some high frequency samples;
        size_t half_output = (output_size / 2);
        size_t diff_size = input_size - output_size;
        memset(samples + half_output, 0, diff_size * sizeof(fft_t));
    } else if (output_size > input_size) {
        size_t half_input = input_size / 2;
        // add some high frequency zero samples
        size_t diff_size = output_size - input_size;
        memmove(samples + half_input + diff_size, samples + half_input, half_input * sizeof(fft_t));
        memset(samples + half_input, 0, diff_size * sizeof(fft_t));
    }
    fft_real_object ifftPlan = fft_real_init(output_size, -1);
    fft_c2r_exec(ifftPlan, samples, output);
    free_real_fft(ifftPlan);
    float norm = 1.0f / input_size;
    for (int i = 0; i < output_size; i++) {
        output[i] = output[i] * norm;
    }
    free(samples);
}

void resampler(std::vector<float> & in, uint32_t sampleRate, std::vector<float>& out, uint32_t out_sampleRate) {    
    uint64_t totalSampleCount = in.size();    
    uint32_t out_size = (uint32_t)(totalSampleCount * ((float) out_sampleRate / sampleRate));
    out.resize(out_size);            
    FFTResample(in.data(),out.data(), totalSampleCount, out_size);    
}
