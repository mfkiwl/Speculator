/**
 *  Signals and Systems Project
 *
 *  @file : main.cu
 *  @author : Dekas Dimitrios
 *  @AEM : 3063
 *  @version : 5.9
 *  @date: Nov 15 12:17AM
 */

/**
 *  Includes
 */
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "AudioFile.h"

/**
 *  Constants
 */
#define A_LOWER_BOUND 11
#define A_RESULT_FILE "./results/convolutionA.txt"
#define C_RESULT_FILE "./results/convolutionC.txt"
#define SAMPLE_AUDIO_FILE_PATH "./resources/sample_audio.wav"
#define PINK_NOISE_FILE_PATH "./resources/pink_noise.wav"
#define PINK_NOISE_SAMPLE_AUDIO_FILE_PATH "./resources/pinkNoise_sampleAudio.wav"
#define WHITE_NOISE_SIZE 441000
#define WHITE_NOISE_SAMPLE_AUDIO_FILE_PATH "./resources/whiteNoise_sampleAudio.wav"

/**
 *  Checks whether a string is an integer or not.
 *
 *  @param str : the string that is going to be checked.
 *  @return a boolean value equal to true if the string is an integer, false otherwise.
 */
bool isInteger(const std::string &str) {
    std::string::const_iterator it = str.begin();    // Create an iterator that points to the string beginning.
    while(it!=str.end() && std::isdigit(*it)) {   // Iterate over the string.
        it++;
    }
    return it==str.end() && !str.empty();    // Return true if the iterator points to the end of the non empty string.
}

/**
 *  Get the user's input and makes sure its an int inside the desired limits.
 *
 *  @return an int equal to the input given by the user.
 */
int getIntInput() {
    int n;
    std::string input;
    std::getline(std::cin, input);
    while (true) {    // While will break as soon as a valid input is given by the user.
        if (isInteger(input)) {
            n = stoi(input);
            if (n >= A_LOWER_BOUND) {
                return n;
            }
        }
        std::cout << "The size of vector A should be an integer greater than " << A_LOWER_BOUND << std::endl;
        std::getline(std::cin, input);
    }
}

/**
 *  Fills a vector with n random double valued numbers varying between 0 and 1.
 *
 *  @param a : a vector we want to fill with random numbers.
 *  @param n : an integer indicating the amount of random numbers we want to fill the vector with.
 */
void fillVector(std::vector<double> &a, const int n) {
    std::random_device rd;
    std::mt19937 rand_engine(rd());
    std::uniform_real_distribution<double> distribution(0, 1);
    for (auto i(0); i < n; i++) {
        a.push_back(distribution(rand_engine));
    }
}

/**
 *  Fills a double array with n random double valued numbers varying between 0 and 1.
 *
 *  @param a : a double array we want to fill with random numbers.
 *  @param n : an integer indicating the amount of random numbers we want to fill the double array with.
 */
void fillArray(double* a, const int n) {
    std::random_device rd;
    std::mt19937 rand_engine(rd());
    std::uniform_real_distribution<double> distribution(0, 1);
    for (auto i(0); i < n; i++) {
        a[i] = distribution(rand_engine);
    }
}

/**
 *  Calculates the convolution of two functions.
 *
 *  @param x : a vector representing the x function on certain values.
 *  @param h : a vector representing the h function on certain values.
 *  @return a vector containing the result of the convolution of the two inputs, x and h.
 */
std::vector<double> myConvolve(const std::vector<double> &x, const std::vector<double> &h) {
    const int xs = x.size();
    const int hs = h.size();
    const int cs = xs + hs - 1; // Size of the convolution's vector
    std::vector<double> c(cs, 0.0);
    for(auto i(0); i < cs; i++) {
        c[i] = 0.0;
        const unsigned long jmin = (i >= xs - 1) ? i - xs + 1 : 0; // The lower bound for the j-loop
        const unsigned long jmax = (i < hs - 1) ? i : hs - 1; // The upper bound for the j-loop
        for(auto j(jmin); j <= jmax; j++) {
            c[i] += x[i - j] * h[j]; // Use convolution's formula
        }
    }
    return c;
}

/**
 *  CUDA Kernel function that calculates the convolution of two signal functions.
 *
 *  @param x : a double array representing the x function on certain values.
 *  @param h : a double array representing the h function on certain values.
 *  @param xs : x array size
 *  @param hs : h array size
 *  @param c : a double array containing the result of the convolution of the two inputs, x and h.
 */
__global__ void cudaConvolve(const double* x, const double* h, const int xs, const int hs, double* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    c[tid] = 0.0;
    const unsigned long jmin = (tid >= xs - 1) ? tid - xs + 1 : 0; // The lower bound for the j-loop
    const unsigned long jmax = (tid < hs - 1) ? tid : hs - 1; // The upper bound for the j-loop
    for(auto j(jmin); j <= jmax; j++) {
        c[tid] += x[tid - j] * h[j];
    }
}

/**
 *  The code used to complete the first task.
 */
void firstTask() {
    // Get the size of vector A
    std::cout << "Enter the wished size of vector A: " << std::endl;
    int size = getIntInput();
    std::vector<double> a;

    // Fill vector A with random numbers
    fillVector(a, size);

    // Define vector B
    std::vector<double> b = {0.2, 0.2, 0.2, 0.2, 0.2};

    // Calculate the convolution between A and B
    std::vector<double> convolution = myConvolve(a, b);

    // Write the results to a txt file
    std::ofstream resFile;
    resFile.open(A_RESULT_FILE);
    if(resFile.fail()) { // Check if the file opened properly
        std::cout << "Could not open the needed file!" << std::endl;
        return;
    }
    for (double c : convolution) {
        resFile << c << "\n";
    }
    resFile.close();
}

/**
 *  The code used to complete the second task.
 */
void secondTask() {
    // Load Files
    AudioFile<double> sampleAudio_file;
    AudioFile<double> pinkNoise_file;
    sampleAudio_file.load(SAMPLE_AUDIO_FILE_PATH);
    pinkNoise_file.load(PINK_NOISE_FILE_PATH);

    // Get Files' Samples
    std::vector<double> sampleAudio_samples = sampleAudio_file.samples.at(0);
    std::vector<double> pinkNoise_samples = pinkNoise_file.samples.at(0);

    // Calculate the samples' convolution
    std::vector<double> pinkNoise_sampleAudio_samples = myConvolve(sampleAudio_samples, pinkNoise_samples);

    // Save the result to a wav file
    AudioFile<double> pinkNoise_sampleAudio_file;
    pinkNoise_sampleAudio_file.samples[0] = pinkNoise_sampleAudio_samples;
    pinkNoise_sampleAudio_file.save(PINK_NOISE_SAMPLE_AUDIO_FILE_PATH, AudioFileFormat::Wave);

    // Define the white noise vector
    std::vector<double> whiteNoise_samples;

    // Generate the white noise signal, a vector containing values between -1 and 1
    fillVector(whiteNoise_samples, WHITE_NOISE_SIZE);
    for (auto i(0); i < WHITE_NOISE_SIZE ; i++) {
        whiteNoise_samples[i] =  (2 * whiteNoise_samples[i]) - 1;
    }

    // Calculate the samples' convolution
    std::vector<double> whiteNoise_sampleAudio_samples = myConvolve(sampleAudio_samples, whiteNoise_samples);

    // Save the result to a wav file
    AudioFile<double> whiteNoise_sampleAudio_file;
    whiteNoise_sampleAudio_file.samples[0] = whiteNoise_sampleAudio_samples;
    whiteNoise_sampleAudio_file.save(WHITE_NOISE_SAMPLE_AUDIO_FILE_PATH, AudioFileFormat::Wave);
}

/**
 *  The code used to complete the third task.
 */
void thirdTask() {
    // Get the size of vector A
    std::cout << "Enter the wished size of vector A in CUDA: " << std::endl;
    int size = getIntInput();
    double* a;
    cudaMallocManaged(&a , size * sizeof(double));

    // Fill A array with random numbers
    fillArray(a, size);

    // Define B array
    int bsize = 5;
    double* b;
    cudaMallocManaged(&b, bsize * sizeof(double));
    for (auto i(0); i < bsize ; ++i) {
        b[i] = 0.2;
    }

    // Define Convolution's array
    int csize = size + bsize - 1;
    double* convolution;
    cudaMallocManaged(&convolution, (size + bsize - 1) * sizeof(double));

    // Calculate the convolution between A and B
    int numThreads = 512;
    int numBlocks = (csize + numThreads - 1) / 512;
    cudaConvolve<<<numBlocks, numThreads>>>(a, b, size, bsize, convolution);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Write the results to a txt file
    std::ofstream resFile;
    resFile.open(C_RESULT_FILE);
    if(resFile.fail()) {
        std::cout << "Could not open the needed file!" << std::endl;
        return;
    }
    for (auto i(0); i < csize; ++i) {
        resFile << convolution[i] << "\n";
    }
    resFile.close();

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(convolution);
}

/**
 *  The main function of the program.
 */
int main() {
    // Task 1 : Simple convolution
    firstTask();
    std::cout << "Simple convolution task completed." << std::endl;
    // Task 2 : WAV Files
    secondTask();
    std::cout << "WAV files task completed." << std::endl;
    // Task 3 : CUDA host code
    thirdTask();
    std::cout << "CUDA task completed." << std::endl;
    return 0;
}