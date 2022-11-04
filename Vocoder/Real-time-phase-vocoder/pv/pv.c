//
//  pv.c
//  
//
//
//

#define true 1
#define false 0

#define FFTSIZE_DEFAULT 2048;
#define RATIO_DEFAULT 1;
#define _USE_MATH_DEFINES

#include "pv.h"
#include "fftext.h"

Phase_vocoder pv_create(void *(*malloc)(size_t), void (*free)(void *))
{
    PV *pv = (PV *)malloc(sizeof(PV));
    pv->malloc = malloc;
    pv->free = free;
    pv->fftsize = FFTSIZE_DEFAULT;
    pv->syn_hopsize = pv->fftsize / 8;
    pv->blocksize = pv->syn_hopsize;
    pv->ratio = RATIO_DEFAULT;
    pv->pre_ratio = RATIO_DEFAULT;
    pv->ana_win = NULL;
    pv->syn_win = NULL;
    pv->input_eff_pos = 0;
    pv->first_time = true;
    pv->initialized = false;
    pv->input_buffer = NULL;
    pv->output_buffer = NULL;
    pv->output = NULL;
    pv->input_frame_head = NULL;
    pv->input_rear = NULL;
    pv->io_pos = NULL;
    pv->count_pointer = NULL;
    pv->ana_frame = NULL;
    pv->syn_frame = NULL;
    pv->realpart = NULL;
    pv->imagpart = NULL;
    pv->mag = NULL;
    pv->ana_phase = NULL;
    pv->syn_phase = NULL;
    pv->pre_ana_phase = NULL;
    pv->pre_syn_phase = NULL;
    pv->bin_freq = NULL;
    pv->phase_increment = NULL;
    pv->estimate_freq = NULL;
    pv->tag_buffer = NULL;
    pv->tag_buffer_head = NULL;
    pv->tag_buffer_rear = NULL;
    
    return (Phase_vocoder)pv;
}

void pv_end(Phase_vocoder x)
{
    PV *pv = (PV *)x;
    fftFree();
    if (pv->ana_win)
        pv->free(pv->ana_win);
    if (pv->syn_win)
        pv->free(pv->syn_win);
    if (pv->input_buffer)
        pv->free(pv->input_buffer);
    if (pv->output_buffer)
        pv->free(pv->output_buffer);
    if (pv->output)
        pv->free(pv->output);
    if (pv->ana_frame)
        pv->free(pv->ana_frame);
    if (pv->syn_frame)
        pv->free(pv->syn_frame);
    if (pv->realpart)
        pv->free(pv->realpart);
    if (pv->imagpart)
        pv->free(pv->imagpart);
    if (pv->mag)
        pv->free(pv->mag);
    if (pv->ana_phase)
        pv->free(pv->ana_phase);
    if (pv->syn_phase)
        pv->free(pv->syn_phase);
    if (pv->pre_ana_phase)
        pv->free(pv->pre_ana_phase);
    if (pv->pre_syn_phase)
        pv->free(pv->pre_syn_phase);
    if (pv->bin_freq)
        pv->free(pv->bin_freq);
    if (pv->phase_increment)
        pv->free(pv->phase_increment);
    if (pv->estimate_freq)
        pv->free(pv->estimate_freq);
    if (pv->tag_buffer)
        pv->free(pv->tag_buffer);
    pv->free(pv);
}

void pv_set_blocksize(Phase_vocoder x, long n)
{
    PV *pv = (PV *)x;
    pv->blocksize = n;
    pv->initialized = false;
}

void pv_set_fftsize(Phase_vocoder x, int n)
{
    PV *pv = (PV *)x;
    pv->fftsize = n;
    pv->initialized = false;
}

void pv_set_ratio(Phase_vocoder x, float ratio)
{
    PV *pv = (PV *)x;
    pv->pre_ratio = pv->ratio;
    pv->ratio = ratio;
}

void pv_set_syn_hopsize(Phase_vocoder x, int n)
{
    PV *pv = (PV *)x;
    pv->syn_hopsize = n;
    pv->initialized = false;
}

void pv_initialize(Phase_vocoder x)
{
    PV *pv = (PV *)x;
    
    //allocate space and initialize for window
    if (! pv->ana_win)
        pv->ana_win = pv_window(pv, hann); //default analysis window is Hanning
    if (! pv->syn_win)
        pv->syn_win = pv_window(pv, hann); //default synthesis window is Hanning
    
    //allocate space and initialize for input buffer and output buffer
    if (pv->input_buffer)
        pv->free(pv->input_buffer);
    long input_buffer_length;
    if (pv->blocksize <= pv->syn_hopsize)
    {
        input_buffer_length = pv->fftsize;
        pv->input_buffer = (float *) pv->malloc(input_buffer_length * 
                                                sizeof(float));
        pv->input_frame_head = pv->input_buffer;
    }
    else
    {
        // the maximum of ana_hopsize is fftsize/3, so the 
        // input_buffer_length is set to the maximum so as to avoid 
        // freeing and allocating memory for input buffer many times 
        // due to the changing of time-stretching ratio.
        
        input_buffer_length = pv->fftsize + 
              round((pv->blocksize / pv->syn_hopsize - 1) * (pv->fftsize / 3));
        
        pv->input_buffer = (float *)pv->malloc(input_buffer_length * 
                                               sizeof(float));
        pv->input_frame_head = pv->input_buffer;
    }
    for (int i = 0; i <= input_buffer_length - 1; i++)
        pv->input_buffer[i] = 0;
    
    if (pv->output_buffer)
        pv->free(pv->output_buffer);
    long output_buffer_length;
    if (pv->blocksize <= pv->syn_hopsize)
    {
        output_buffer_length = pv->fftsize;
        pv->output_buffer = (float *)pv->malloc(output_buffer_length * 
                                                sizeof(float));
        pv->io_pos = pv->output_buffer + pv->syn_hopsize;
    }
    else
    {
        output_buffer_length = pv->fftsize + pv->blocksize - pv->syn_hopsize;
        pv->output_buffer = (float *)pv->malloc(output_buffer_length * 
                                                sizeof(float));
        pv->io_pos = pv->output_buffer + pv->blocksize;
    }
    pv->count_pointer = pv->output_buffer;
    for (int i = 0; i <= output_buffer_length - 1; i++)
        pv->output_buffer[i] = 0;
    
    if (pv->output)
        pv->free(pv->output);
    pv->output = (float *)pv->malloc(pv->blocksize * sizeof(float));
    
    if (pv->ana_frame)
        pv->free(pv->ana_frame);
    pv->ana_frame = (float *)pv->malloc(pv->fftsize * sizeof(float));
    if (pv->syn_frame)
        pv->free(pv->syn_frame);
    pv->syn_frame = (float *)pv->malloc(pv->fftsize * sizeof(float));
    
    if (pv->realpart)
        pv->free(pv->realpart);
    pv->realpart = (float *)pv->malloc((pv->fftsize / 2 + 1) * sizeof(float));
    if (pv->imagpart)
        pv->free(pv->imagpart);
    pv->imagpart = (float *)pv->malloc((pv->fftsize / 2 + 1) * sizeof(float));
    if (pv->mag)
        pv->free(pv->mag);
    pv->mag = (float *)pv->malloc((pv->fftsize / 2 + 1) * sizeof(float));
    
    // allocate space for phase and pre_phase which will be used in 
    // phase unwrapping
    if (pv->ana_phase)
        pv->free(pv->ana_phase);
    pv->ana_phase = (float *)pv->malloc((pv->fftsize / 2 + 1) * sizeof(float));
    
    if (pv->syn_phase)
        pv->free(pv->syn_phase);
    pv->syn_phase = (float *)pv->malloc((pv->fftsize / 2 + 1) * sizeof(float));
    
    if (pv->pre_ana_phase)
        pv->free(pv->pre_ana_phase);
    pv->pre_ana_phase = (float *)pv->malloc((pv->fftsize / 2 + 1) * 
                                            sizeof(float));
    
    if (pv->pre_syn_phase)
        pv->free(pv->pre_syn_phase);
    pv->pre_syn_phase = (float *)pv->malloc((pv->fftsize / 2 + 1) * 
                                            sizeof(float));
    
    // bin frequency, used in phase unwrapping
    if (pv->bin_freq)
        pv->free(pv->bin_freq);
    pv->bin_freq = (float *)pv->malloc((pv->fftsize / 2 + 1) * sizeof(float));
    for (int i = 0; i <= pv->fftsize / 2; i++)
        pv->bin_freq[i] = 2 * (M_PI) * i / pv->fftsize;
    
    if (pv->phase_increment)
        pv->free(pv->phase_increment);
    pv->phase_increment = (float *)pv->malloc((pv->fftsize / 2 + 1) * 
                                              sizeof(float));
    
    if (pv->estimate_freq)
        pv->free(pv->estimate_freq);
    pv->estimate_freq = (float *)pv->malloc((pv->fftsize / 2 + 1) * 
                                            sizeof(float));
    
    // circular tag buffer for servomechanism
    if (pv->tag_buffer)
        pv->free(pv->tag_buffer);
    if (pv->blocksize <= pv->syn_hopsize)
    {
        pv->queue_length = pv->fftsize / (pv->syn_hopsize * 2) + 2;
        pv->tag_buffer = (struct tag *)pv->malloc((pv->queue_length + 1)
                                                 * sizeof(struct tag));
    }
    else
    {
        pv->queue_length = pv->fftsize / (pv->syn_hopsize * 2)
                           + (2 * pv->blocksize) / pv->syn_hopsize;
        pv->tag_buffer = (struct tag *)pv->malloc((pv->queue_length + 1)
                                                  * sizeof(struct tag));
    }
    
    pv->tag_buffer_head = pv->tag_buffer;
    pv->tag_buffer_rear = pv->tag_buffer;

    pv->initialized = true;
}

void pv_set_ana_window(Phase_vocoder x, float *window)
{
    PV *pv = (PV*)x;
    if (pv->ana_win)
        pv->free(pv->ana_win);
    pv->ana_win = (float *)pv->malloc(pv->fftsize * sizeof(float));
    memcpy(pv->ana_win, window, pv->fftsize * sizeof(float));
}

void pv_set_syn_window(Phase_vocoder x, float *window)
{
    PV *pv = (PV*)x;
    if (pv->syn_win)
        pv->free(pv->syn_win);
    pv->syn_win = (float *)pv->malloc(pv->fftsize * sizeof(float));
    memcpy(pv->syn_win, window, pv->fftsize * sizeof(float));
}

float *pv_window(Phase_vocoder x, float (*window_type)(double x)) 
    // window is after normalized
{
    PV *pv = (PV *)x;
    float sum_window_square = 0, COLA_factor;
    int window_length = pv->fftsize;
    float *window = (float *)pv->malloc(window_length * sizeof(float));
    for (int i = 0; i <= window_length - 1; i++)
    {
        window[i] = window_type((double)i / window_length);
        // note that the computation is all double even if window[i] is float
        sum_window_square += window[i] * window[i];
    }
    COLA_factor = sum_window_square / pv->syn_hopsize;
    for (int i = 0; i <= pv->fftsize - 1; i++)
        window[i] = window[i] / sqrt(COLA_factor);
    return window;
}

long pv_get_input_count(Phase_vocoder x)
{
    PV *pv = (PV*)x;
    long input_count;
    int ana_hopsize = round((pv->syn_hopsize) * (pv->ratio));
    
    if (! (pv->initialized))
        pv_initialize(x);
    
    if (pv->blocksize <= pv->syn_hopsize)
    {
        if (pv->first_time)
            input_count = pv->fftsize;
        else if (pv->count_pointer == pv->output_buffer)
            input_count = ana_hopsize;
        else
            input_count = 0;
    }
    else
    {
        if (pv->first_time)
            input_count = pv->fftsize + (pv->blocksize / pv->syn_hopsize - 1) *
                                        ana_hopsize;
        else
            input_count = pv->blocksize / pv->syn_hopsize * ana_hopsize;
    }
    
    return input_count;
}

float pv_get_effective_pos(Phase_vocoder x, long sample_number)
{
    PV *pv = (PV*)x;
    if (! (pv->initialized))
        pv_initialize(x);
    
    struct tag *tag_find = NULL; // move on the queue to find the appropriate tag for
                                 // the computation of effective audio position;
    if (pv->tag_buffer_head == pv->tag_buffer + pv->queue_length)
        tag_find = pv->tag_buffer;
    else
        tag_find = pv->tag_buffer_head + 1;
    
    if (sample_number > (* pv->tag_buffer_rear).syn_tag)
    {
        printf("Error pv_get_effective_pos(): Invalid Input Sample Number!\n");
        return 0;
    } // avoid deliberately use a really big sample_number as input
    else
    {
        while (sample_number > (*tag_find).syn_tag)
        {
            if (tag_find == pv->tag_buffer + pv->queue_length)
                tag_find = pv->tag_buffer;
            else
                tag_find += 1;
        }
        if ((*tag_find).syn_tag == sample_number)
        {
            if (tag_find == pv->tag_buffer)
                pv->tag_buffer_head = pv->tag_buffer + pv->queue_length;
            else
                pv->tag_buffer_head = tag_find - 1;
            return (*tag_find).ana_tag;
        }
        else
        {
            struct tag *tag_pre_find;
            if (tag_find == pv->tag_buffer_head)
                tag_pre_find = pv->tag_buffer + pv->queue_length;
            else
                tag_pre_find = tag_find - 1;
        
            if (tag_pre_find == pv->tag_buffer_head)
                return (float)sample_number;
            else
            {
                if (tag_pre_find == pv->tag_buffer)
                    pv->tag_buffer_head = pv->tag_buffer + pv->queue_length;
                else
                    pv->tag_buffer_head = tag_pre_find - 1;
            
                double interval = ((sample_number - (*tag_pre_find).syn_tag))
                           / ((*tag_find).syn_tag - (*tag_pre_find).syn_tag);
                return interval * ( (*tag_find).ana_tag - (*tag_pre_find).ana_tag )
                           + (*tag_pre_find).ana_tag;
            }
        }
    }
}

void pv_put_input(Phase_vocoder x, long size, float *samples)
    // 'samples' points to samples to be sent each time
{
    if (size > 0)
    {
        PV *pv = (PV*)x;
        if (! (pv->initialized))
            pv_initialize(x);
        long pre_valid_input_length, valid_input_length;
        int ana_hopsize = round(pv->syn_hopsize * pv->ratio);
        int pre_ana_hopsize = round(pv->syn_hopsize * pv->pre_ratio);
        if (pv->blocksize <= pv->syn_hopsize)
        {
            valid_input_length = pv->fftsize;
            pre_valid_input_length = pv->fftsize;
        }
        else
        {
            valid_input_length = pv->fftsize + 
                    (pv->blocksize / pv->syn_hopsize - 1) * ana_hopsize;
            pre_valid_input_length = pv->fftsize + 
                    (pv->blocksize / pv->syn_hopsize - 1) * pre_ana_hopsize;
        }
    
        // NOTICE: for first frame, the 'size' must be the valid_input_length,
        // or the following code would result in fault
        //
        // copy the same sample points from last frame:
        for (int i = 0; i <= valid_input_length - size - 1; i++) 
            pv->input_buffer[i] = pv->input_buffer[i + pre_valid_input_length -
                                                   (valid_input_length - size)];
        // get new sample points from outside:
        for (int i = 0; i <= size - 1; i++)
            pv->input_buffer[valid_input_length - size + i] = samples[i];
    
        // input_rear initialzation
        pv->input_rear = pv->input_buffer + valid_input_length - 1;
        
        // input_frame_head initialization
        pv->input_frame_head = pv->input_buffer;
    }
}

float *pv_get_output(Phase_vocoder x)
{
    PV *pv = (PV *)x;
    if (! (pv->initialized))
        pv_initialize(x);
    
    long blocksize = pv->blocksize;
    int fftsize = pv->fftsize;
    int syn_hopsize = pv->syn_hopsize;
    float ratio = pv->ratio;
    bool first_time = pv->first_time;
    float *ana_win = pv->ana_win;
    float *syn_win = pv->syn_win;
    float *input_buffer =  pv->input_buffer;
    float *output_buffer = pv->output_buffer;
    float *output = pv->output;
    float *input_frame_head = pv->input_frame_head;
    float *input_rear = pv->input_rear;
    float *io_pos = pv->io_pos;
    float *count_pointer = pv->count_pointer;
    float *ana_frame = pv->ana_frame;
    float *syn_frame = pv->syn_frame;
    float *realpart = pv->realpart;
    float *imagpart = pv->imagpart;
    float *mag = pv->mag;
    float *ana_phase = pv->ana_phase;
    float *syn_phase = pv->syn_phase;
    float *pre_ana_phase = pv->pre_ana_phase;
    float *pre_syn_phase = pv->pre_syn_phase;
    float *bin_freq = pv->bin_freq;
    float *phase_increment = pv->phase_increment;
    float *estimate_freq = pv->estimate_freq;
    
    long temp_ana_tag;  // store the tag of the last frame, used to compute
                        // the tag of next frame
    long temp_syn_tag;
    
    int ana_hopsize = round(syn_hopsize * ratio);
    
    int M = (int) ( log((double)fftsize) / log((double)2) ); // used in FFT
    int fft_error_sign, ifft_error_sign;
    
    while ((input_frame_head + fftsize - 1 <= input_rear) && (count_pointer < io_pos))
    {
        for (int i = 0; i <= fftsize - 1; i++) // get and window the buffer
            ana_frame[i] = input_frame_head[i] * ana_win[i];

        OneDimensionFFTshift(ana_frame, fftsize);  // FFTshift
        
        fft_error_sign = fftInit(M);		// FFT
        rffts(ana_frame, M, 1);
        
        /* get magnitude and phase */
        // modify the order of the output of FFT, make sure the order is
        // frequency increase
        realpart[0] = ana_frame[0];
        imagpart[0] = 0;
        mag[0] = (float)sqrt((realpart[0]) * (realpart[0]) + 
                             (imagpart[0]) * (imagpart[0]));
        ana_phase[0] = (float)atan2(imagpart[0], realpart[0]);
        realpart[fftsize / 2] = ana_frame[1];
        imagpart[fftsize / 2] = 0;
        mag[fftsize / 2] = (float)sqrt(
                (realpart[fftsize / 2]) * (realpart[fftsize / 2]) + 
                (imagpart[fftsize / 2]) * (imagpart[fftsize / 2]));
        ana_phase[fftsize / 2] = (float)atan2(
                imagpart[fftsize / 2], realpart[fftsize / 2]);
        for (int i = 1; i <= fftsize / 2 - 1; i++)
        {
            realpart[i] = ana_frame[2 * i];
            imagpart[i] = ana_frame[2 * i + 1];
            mag[i] = (float)sqrt((realpart[i]) * (realpart[i]) +
                                 (imagpart[i]) * (imagpart[i]));
            ana_phase[i] = (float)atan2(imagpart[i], realpart[i]);
        }
        
        /* phase unwrapping & set synthesis phase */
        for (int i = 0; i <= fftsize / 2; i++)
        {
            if (first_time)
            {
                syn_phase[i] = ana_phase[i];
                first_time = false;
            }
            else
            {
                phase_increment[i] = ana_phase[i] - pre_ana_phase[i] -
                                     bin_freq[i] * ana_hopsize;
                
                while (phase_increment[i] > M_PI)
                    phase_increment[i] -= (float)(2 * M_PI);
                while (phase_increment[i] < -M_PI)
                    phase_increment[i] += (float)(2 * M_PI);
                estimate_freq[i] = phase_increment[i] / ana_hopsize +
                                   bin_freq[i];
                
                // set synthesis phase
                syn_phase[i] = pre_syn_phase[i] + 
                               syn_hopsize * estimate_freq[i];
            }
            
            // record phases
            pre_ana_phase[i] = ana_phase[i];
            pre_syn_phase[i] = syn_phase[i];
            
            // update realpart and imagpart
            realpart[i] = mag[i] * cos(syn_phase[i]);
            imagpart[i] = mag[i] * sin(syn_phase[i]);
        }
        
        // iFFT buffer; recover the order of the buffer to get ready for iFFT
        for (int i = 0; i <= (fftsize / 2) - 1; i++)
            syn_frame[i * 2] = realpart[i];
        syn_frame[1] = realpart[fftsize / 2];
        for (int i = 1; i <= (fftsize / 2) - 1; i++)
            syn_frame[i * 2 + 1] = imagpart[i];
        
        // inverse FFT
        // it's OK if we don't call fftInit() here, as fft size is the
        //     same as the first call
        ifft_error_sign = fftInit(M);
        riffts(syn_frame, M, 1);
        
        // fftshift
        OneDimensionFFTshift(syn_frame, fftsize);
        
        // window the frame and then add it to the output buffer
        for (int i = 0; i <= fftsize - 1; i++)
        {
            syn_frame[i] = syn_win[i] * syn_frame[i];
            count_pointer[i] = count_pointer[i] + syn_frame[i];
        }
        
        input_frame_head += ana_hopsize;
        
        if (blocksize > syn_hopsize)
            count_pointer += syn_hopsize;
        else
            break;
    }
    
    for (int i = 0; i <= blocksize - 1; i++) // output synthesized samples
        output[i] = output_buffer[i];
    
    // modify the synthesis buffer to get ready for next time synthesis
    if (blocksize <= syn_hopsize)
    {
        if (count_pointer + blocksize < io_pos)
            count_pointer += blocksize;
        else
            count_pointer = output_buffer;
        
        for (int i = 0; i <= fftsize - blocksize - 1; i++)
            output_buffer[i] = output_buffer[i + blocksize];
        for (long i = fftsize - blocksize; i <= fftsize - 1; i++)
            output_buffer[i] = 0;
    }
    else
    {
        for (int i = 0; i <= fftsize - syn_hopsize - 1; i++)
            output_buffer[i] = output_buffer[i + blocksize];
        for (int i = fftsize - syn_hopsize; 
             i <= fftsize + blocksize - syn_hopsize - 1; i++)
            output_buffer[i] = 0;
        count_pointer = output_buffer;
    }
    
    // Put the tag of the processed frame into the queue
    if (pv->tag_buffer_rear == pv->tag_buffer + pv->queue_length)
        pv->tag_buffer_rear = pv->tag_buffer;
    else
        pv->tag_buffer_rear += 1;
    
    if (pv->tag_buffer_head == pv->tag_buffer_rear) // queue full
    {
        long sample_number;  //TODO:sample number should be get from audio player
        pv_get_effective_pos(pv, sample_number);
        // call pv_get_effective_pos() to clear old tags in the queue
    }

    if (first_time)
    {
        ( *(pv->tag_buffer_rear) ).ana_tag = (pv->fftsize) / 2;
        ( *(pv->tag_buffer_rear) ).syn_tag = (pv->fftsize) / 2;
    }
    else
    {
        ( *(pv->tag_buffer_rear) ).ana_tag = ana_hopsize + temp_ana_tag;
        ( *(pv->tag_buffer_rear) ).syn_tag = syn_hopsize + temp_syn_tag;
    }
    temp_ana_tag = ( *(pv->tag_buffer_rear) ).ana_tag;
    temp_syn_tag = ( *(pv->tag_buffer_rear) ).syn_tag;

    return output;
}