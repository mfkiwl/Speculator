#ifndef BPM_DETECTOR_H
#define BPM_DETECTOR_H

int most_frequent_bpm(int *);

int max_array(float *, int);

float *comb_filter_convolution(float *data_input, float *energy,
                               unsigned int N, unsigned int sample_rate, float resolution,
                               int minbpm, int maxbpm, int high_limit);

float *differentiator(float *input_buffer, unsigned int N);

float *half_wave_rectifier(float *input_buffer, unsigned int N);

float *full_wave_rectifier(float *input_buffer, unsigned int N);

float *hanning_window(float *data_in, unsigned int N, unsigned int sampling_rate);

float **filterbank(float *time_data_in, float **filterbank, unsigned int N, unsigned int sampling_rate);

float compute_bpm(float *energy_buffer, unsigned int bpm_range, unsigned int minbpm, float resolution);

float *clear_energy_buffer(float *energy_buffer, unsigned int bpm_range, float resolution);

#endif // BPMDETECTOR_BPM_DETECTOR_H
