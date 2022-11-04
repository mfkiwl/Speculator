
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "kiss_fft/kiss_fftr.h"
#include "bpm_detector.h"
#include <stdint.h>
//ref : https://github.com/mackron/dr_libs/blob/master/dr_wav.h
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "timing.h"

int16_t *wavRead_int16(char *filename, uint32_t *sampleRate, uint64_t *totalSampleCount) {
    unsigned int channels;
    int16_t *buffer = drwav_open_and_read_file_s16(filename, &channels, sampleRate, totalSampleCount);
    if (buffer == NULL) {
        printf("ERROR.");
    }
    return buffer;
}

int main(int argc, char **argv) {
    if (argc < 2) /* argc should be 2 for correct execution */
    {
        printf("usage: %s relative filename\n", argv[0]);
        return -1;
    }

    char *filename = argv[1];
    uint32_t sample_rate = 0;
    uint64_t num_samples = 0;
    int16_t *data_in = wavRead_int16(filename, &sample_rate, &num_samples);
    if (data_in == 0) return -1;

    // the valid amplitude range for values based on the bits per sample
    int low_limit = -32768;
    int high_limit = 32767;
    double startTime = now();
    printf("Valid range for data values : %d to %d \n", low_limit, high_limit);
    /***********************************************************************************/
    // Start reading audio data and computing bpm
    /***********************************************************************************/

    unsigned int i = 0, j = 0, k = 0;

    // Number of samples to analyze. We are taking 4 seconds of data.
    unsigned int N = 4 * sample_rate;
    if (N % 2 != 0)
        N += 1;
    // Used to recursively call comb filter bpm detection
    float a, b, c, d;
    int minbpm = 60;
    int maxbpm = 180;
    int bpm_range = maxbpm - minbpm;
    int resolution = 1;
    float current_bpm = 0;
    int winning_bpm = 0;

    int *frequency_map = (int *) calloc(200, sizeof(int));
    // Allocate history buffer to hold energy values for each tested BPM
    float *energy = (float *) malloc(sizeof(float) * (bpm_range / resolution));
    // Allocate FFT buffers to read in data
    float *fft_input = (float *) malloc(N * sizeof(float));
    // Allocate subband buffers
    size_t num_sub_bands = 6;
    unsigned int sub_band_size = N / num_sub_bands;
    float **sub_band_input = (float **) calloc(num_sub_bands, sizeof(float *));
    for (i = 0; i < num_sub_bands; i++) {
        sub_band_input[i] = (float *) calloc(N, sizeof(float));
    }  // Read in data from file to left and right channel buffers
    for (i = 0; i < N; i++) {
        // Set data to FFT buffers
        fft_input[i] = (float) data_in[i];
    } // End read in data
    // Split data into separate frequency bands
    sub_band_input = filterbank(fft_input, sub_band_input, N, sample_rate);
    // Clear energy buffer before calculating new energy data
    energy = clear_energy_buffer(energy, bpm_range, resolution);
    // Filter and sum energy for each tested bpm for each subband
    for (i = 0; i < num_sub_bands; i++) {
        sub_band_input[i] = full_wave_rectifier(sub_band_input[i], N);
        sub_band_input[i] = hanning_window(sub_band_input[i], N, sample_rate);
        sub_band_input[i] = differentiator(sub_band_input[i], N);
        sub_band_input[i] = half_wave_rectifier(sub_band_input[i], N);
        energy = comb_filter_convolution(sub_band_input[i], energy, N, sample_rate,
                                         1, minbpm, maxbpm, high_limit);

    }
    // Calculate the bpm from the total energy
    current_bpm = compute_bpm(energy, bpm_range, minbpm, resolution);
    if (current_bpm != -1) {
        frequency_map[(int) roundf(current_bpm)] += 1;
    }
    winning_bpm = most_frequent_bpm(frequency_map);
    double time_interval = calcElapsed(startTime, now());
    printf("BPM winner is: %i\n", winning_bpm);

    for (i = 0; i < num_sub_bands; i++) {
        free(sub_band_input[i]);
    }
    free(sub_band_input);
    free(fft_input);
    free(data_in);
    // Finish timing program
    printf("time interval: %d ms\n ", (int) (time_interval * 1000));
}
/*
 * Split time domain data in data_in into 6 separate frequency bands
 *   Band limits are [0 200 400 800 1600 3200]
 *
 * Output a vector of 6 time domain arrays of size N
 */
float **filterbank(float *time_data_in, float **filterbank,
                   unsigned int N, unsigned int sampling_rate) {

    // Initialize array of bandlimits
    int *bandlimits, *bandleft, *bandright;
    bandleft = (int *) malloc(sizeof(int) * 6);
    bandright = (int *) malloc(sizeof(int) * 6);
    bandlimits = (int *) malloc(sizeof(int) * 6);
    bandlimits[0] = 3;
    bandlimits[1] = 200;
    bandlimits[2] = 400;
    bandlimits[3] = 800;
    bandlimits[4] = 1600;
    bandlimits[5] = 3200;

    // Compute the boundaries of the bandlimits in terms of array location
    int i, j, maxfreq;
    maxfreq = sampling_rate / 2;
    for (i = 0; i < 5; i++) {
        bandleft[i] = floor(bandlimits[i] * N / (2 * maxfreq)) + 1;
        bandright[i] = floor(bandlimits[i + 1] * N / (2 * maxfreq));
    }
    bandleft[5] = floor(bandlimits[5] / maxfreq * N / 2) + 1;
    bandright[5] = floor(N / 2);

    // Initialize FFT buffers
    kiss_fftr_cfg fft_cfg = kiss_fftr_alloc(N, 0, NULL, NULL);
    kiss_fftr_cfg fft_inv_cfg = kiss_fftr_alloc(N, 1, NULL, NULL);
    kiss_fft_cpx *freq_data_in, *freq_data_out;
    freq_data_in = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
    freq_data_out = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));

    // Take FFT of input time domain data
    kiss_fftr(fft_cfg, time_data_in, freq_data_in);

    for (i = 0; i < 6; i++) {
        memset(freq_data_out, 0, N * sizeof(kiss_fft_cpx));
        for (j = bandleft[i]; j < bandright[i]; j++) {
            freq_data_out[j].r = freq_data_in[j].r;
            freq_data_out[j].i = freq_data_in[j].i;
            freq_data_out[N - j].r = freq_data_in[N - j].r;
            freq_data_out[N - j].i = freq_data_in[N - j].i;
        }
        kiss_fftri(fft_inv_cfg, freq_data_out, filterbank[i]);
    }
    free(bandlimits);
    free(bandleft);
    free(bandright);
    free(freq_data_in);
    free(freq_data_out);
    return filterbank;
}

/*
  * Implement a 200 ms half hanning window.
  *
  * Input is a signal in the frequency domain
  * Output is a windowed signal in the frequency domain.
  */
float *hanning_window(float *data_in, unsigned int N, unsigned int sampling_rate) {

    kiss_fftr_cfg fft_window_cfg = kiss_fftr_alloc(N, 0, NULL, NULL);
    kiss_fftr_cfg fft_data_cfg = kiss_fftr_alloc(N, 0, NULL, NULL);
    kiss_fftr_cfg fft_data_inv_cfg = kiss_fftr_alloc(N, 1, NULL, NULL);
    float *hanning_in;
    kiss_fft_cpx *hanning_out, *data_out, *temp_data;
    hanning_in = (float *) malloc(N * sizeof(float));
    hanning_out = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
    data_out = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
    temp_data = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
    int hann_len = .2f * sampling_rate;
    int i;
    for (i = 0; i < N; i++) {
        if (i < hann_len) {
            hanning_in[i] = pow(cosf(2 * i * M_PI / hann_len), 2);
        } else
            hanning_in[i] = 0.0f;
        hanning_out[i].r = 0.0f;
        hanning_out[i].i = 0.0f;
    }
    hanning_in[0] = 0.0f;

    kiss_fftr(fft_window_cfg, hanning_in, hanning_out);
    kiss_fftr(fft_data_cfg, data_in, data_out);

    for (i = 0; i < N; i++) {
        temp_data[i].r = data_out[i].r * hanning_out[i].r - data_out[i].i * hanning_out[i].i;
        temp_data[i].i = data_out[i].i * hanning_out[i].r + data_out[i].r * hanning_out[i].i;
    }

    kiss_fftri(fft_data_inv_cfg, temp_data, data_in);

    free(hanning_in);
    free(hanning_out);
    free(data_out);
    free(temp_data);
    kiss_fft_cleanup();

    return data_in;
}

/*
    * Rectifies a signal in the time domain.
    */
float *full_wave_rectifier(float *input_buffer, unsigned int N) {

    for (int i = 1; i < N; i++) {
        if (input_buffer[i] < 0.0f)
            input_buffer[i] = -input_buffer[i];
    }
    return input_buffer;
}

/*
  * Rectifies a signal in the time domain.
  */
float *half_wave_rectifier(float *input_buffer, unsigned int N) {

    for (int i = 1; i < N; i++) {
        if (input_buffer[i] < 0.0f)
            input_buffer[i] = 0.0f;
    }

    return input_buffer;
}

/*
 * Differentiates a signal in the time domain.
 */
float *differentiator(float *input_buffer, unsigned int N) {

    float prev = input_buffer[0];
    input_buffer[0] = 0;
    float out = 0;
    for (int i = 1; i < N; i++) {
        out = input_buffer[i] - prev;
        prev = input_buffer[i];
        input_buffer[i] = out;
    }
    return input_buffer;
}

float *comb_filter_convolution(float *data_input, float *energy,
                               unsigned int N, unsigned int sample_rate, float resolution,
                               int minbpm, int maxbpm, int high_limit) {
    /*
     * Convolves the FFT of the data_input of size N with an impulse train
     * with a periodicity relative to the bpm in the range of minbpm to maxbpm.
     *
     * Returns energy array
     */
    kiss_fftr_cfg fft_cfg_filter = kiss_fftr_alloc(N, 0, NULL, NULL);
    kiss_fftr_cfg fft_cfg_data = kiss_fftr_alloc(N, 0, NULL, NULL);

    float *filter_input;
    filter_input = (float *) malloc(N * sizeof(kiss_fft_cpx));

    kiss_fft_cpx *filter_output, *data_output;
    filter_output = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
    data_output = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));

    kiss_fftr(fft_cfg_data, data_input, data_output);

    int id;
    float i, a;
    unsigned int j, ti;
    float temp_energy_r, temp_energy_i;
    unsigned int bpm_range = maxbpm - minbpm;

    for (i = 0; i < bpm_range; (i += resolution)) {

        // Ti is the period of impulses (samples per beat)
        ti = floor((float) 60 / (minbpm + i) * sample_rate);

        for (j = 0; j < N; j++) {

            if (j % ti == 0) {
                filter_input[j] = (float) high_limit;
            } else {
                filter_input[j] = 0.0f;
            }
        }
        kiss_fftr(fft_cfg_filter, filter_input, filter_output);

        id = floor(i / resolution);

        for (j = 0; j < N; j++) {
            a = pow(.5f, j / ti);
            a *= (60 + .1f * i) / maxbpm;
            temp_energy_r = (filter_output[j].r * data_output[j].r - filter_output[j].i * data_output[j].i);
            temp_energy_i = (filter_output[j].r * data_output[j].i + filter_output[j].i * data_output[j].r);

            energy[id] += pow(pow(temp_energy_i, 2) + pow(temp_energy_r, 2), .5f) * a;
        }
        //printf("Energy of bpm %f is %f\n", (minbpm+i), energy[id]);
    }

    free(filter_input);
    free(filter_output);
    free(data_output);
    kiss_fft_cleanup();

    return energy;
}

float compute_bpm(float *energy_buffer, unsigned int bpm_range, unsigned int minbpm, float resolution) {
    float bpm;
    bpm = (float) max_array(energy_buffer, bpm_range / resolution);
    if (bpm != -1) {
        bpm *= resolution;
        bpm += minbpm;
    }
    return bpm;

}

float *clear_energy_buffer(float *energy_buffer, unsigned int bpm_range, float resolution) {
    for (int i = 0; i < bpm_range; (i += resolution)) {
        int id = i / resolution;
        energy_buffer[id] = 0.0f;
    }
    return energy_buffer;
}

/*
   * Computes the max element of the array
   * and returns the corresponding index.
   */
int max_array(float *array, int size) {

    int i, index = 0;
    float max = 0;
    for (i = 0; i < size; i++) {
        if (array[i] > max) {
            max = array[i];
            index = i;
        }
    }
    if (max == 0.0f) return -1;
    else
        return index;
}

int most_frequent_bpm(int *map) {
    int i, winner = 0, value = 0;
    for (i = 0; i < 200; i++) {
        if (map[i] > value) {
            winner = i;
            value = map[winner];
        }
    }
    return winner;
}
