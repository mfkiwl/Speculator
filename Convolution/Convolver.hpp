//https://github.com/tmtakashi/3d-guitar-amp-renderer
#pragma once
#include "fftw3.h"
#include <complex>
#include <vector>

constexpr int floorMod(int a, int b) { return (b + (a % b)) % b; }

class Convolver
{
  public:
    enum class ConvolutionMethod
    {
        OverlapAdd,
        OverlapSave,
        UniformOLS,
    };
    Convolver(Convolver::ConvolutionMethod convolutionMethod);
    void prepare(int bufSize, int fftSz, int irSz,
                 std::complex<float> *newIR) noexcept;
    void prepare(int bufSize, int numParts,
                 std::complex<float> *partIRs) noexcept;
    void process(const float *in, float *out);
    // IR size need to be the same to call this function
    void setNewIR(std::complex<float> *newIR);
    void setNewPartitionedIR(std::complex<float> *newIR);

  private:
    int bufferSize = 0;
    int fftSize = 0;
    int irSize = 0;

    // pre-computed rfft array of the impulse response(size: fftSize / 2 + 1)
    std::complex<float> *IR;
    // pre-computed rfft array of the uniformly partitioned impulse response
    std::complex<float> *partitionedIR;
    std::size_t numPartitions;

    std::vector<std::complex<float>> fdlRegister;
    int fdlRegisterSize = 0;
    int fdlRegisterPointer = 0;

    // rfft array container for input buffer
    std::vector<std::complex<float>> IN;
    // array container for frequency domain
    std::vector<std::complex<float>> Y;
    // array container for convolution result
    std::vector<float> filteredInput;
    std::vector<float> paddedIn;

    // array container for overlap add
    std::vector<float> overlapAddBuffer;
    int overlapAddBufferSize = 0;
    int overlapAddFrontPointer = 0;

    // array container for overlap save
    std::vector<float> overlapSaveBuffer;

    int overlapMask = 0;

    void uniformOLSProcess(const float *in, float *out);
    Convolver::ConvolutionMethod method;

    void fftConvolution(float *in, float *out, std::size_t fftSize);
};

Convolver::Convolver(Convolver::ConvolutionMethod convolutionMethod)
    : method(convolutionMethod)
{
}

void Convolver::prepare(int bufSize, int irSz, int fftSz,
                        std::complex<float> *newIR) noexcept
{
    // set parameters
    bufferSize = bufSize;
    irSize = irSz;
    ;
    // needs to be 2 * bufferSize for UniformOLS
    fftSize = fftSz;
    if (method == ConvolutionMethod::OverlapAdd)
    {
        overlapAddBufferSize = std::pow(
            2, std::ceil(std::log(bufferSize + irSize - 1) / std::log(2)));

        overlapAddBuffer.resize(overlapAddBufferSize, 0);
        paddedIn.resize(fftSize, 0.0f);
        filteredInput.resize(fftSize);
        IN.resize(fftSize / 2 + 1);
        Y.resize(fftSize / 2 + 1);
    }
    else if (method == ConvolutionMethod::OverlapSave)
    {
        overlapSaveBuffer.resize(fftSize, 0);
        paddedIn.resize(fftSize, 0.0f);
        filteredInput.resize(fftSize);
        IN.resize(fftSize / 2 + 1);
        Y.resize(fftSize / 2 + 1);
    }

    // assign pointer to pre-computed impulse response rFFT
    IR = newIR;

    overlapMask = overlapAddBufferSize - 1;
}
void Convolver::prepare(int bufSize, int numParts,
                        std::complex<float> *partIRs) noexcept
{
    // set parameters
    bufferSize = bufSize;
    ;
    numPartitions = numParts;
    if (method == ConvolutionMethod::UniformOLS)
    {
        overlapSaveBuffer.resize(2 * bufferSize, 0.0f);
        paddedIn.resize(2 * bufferSize, 0.0f);
        filteredInput.resize(2 * bufferSize);
        IN.resize(bufferSize + 1);
        Y.resize(bufferSize + 1, std::complex<float>{0.0f, 0.0f});
        fdlRegister.resize((bufferSize + 1) * numPartitions,
                           std::complex<float>{0.0f, 0.0f});
        fdlRegisterPointer = (bufferSize + 1) * (numPartitions - 1);
        fdlRegisterSize = fdlRegister.size();
    }

    // assign pointer to pre-computed impulse response rFFT
    partitionedIR = partIRs;
}

void Convolver::process(const float *in, float *out)
{
    switch (method)
    {
    case ConvolutionMethod::OverlapAdd:
        memcpy(paddedIn.data(), in, bufferSize * sizeof(float)); // zero-padding

        fftConvolution(paddedIn.data(), filteredInput.data(), fftSize);

        // calculate output buffer
        for (int i = 0; i < bufferSize; ++i)
        {
            out[i] = 0.3 * (filteredInput[i] / fftSize +
                            overlapAddBuffer[overlapAddFrontPointer]);
            ++overlapAddFrontPointer;
            overlapAddFrontPointer &= overlapMask; // wire-anding
        }

        // update overlap buffer
        if (overlapAddFrontPointer - bufferSize >= 0)
        {
            memset(&overlapAddBuffer[overlapAddFrontPointer] - bufferSize, 0,
                   bufferSize * sizeof(float));
        }
        else
        {
            memset(overlapAddBuffer.data(), 0,
                   overlapAddFrontPointer *
                       sizeof(float)); // fill front of the buffer
            memset(&overlapAddBuffer[overlapAddBufferSize -
                                     (bufferSize - overlapAddFrontPointer)],
                   0,
                   (bufferSize - overlapAddFrontPointer) *
                       sizeof(float)); // fill tail of the buffer
        }
        for (int i = 0; i < irSize - 1; ++i)
        {
            overlapAddBuffer[(overlapAddFrontPointer + i) & overlapMask] +=
                filteredInput[bufferSize + i] / fftSize;
        }
        break;
    case ConvolutionMethod::OverlapSave:
        // copy new input to the tail of buffer
        memcpy(&overlapSaveBuffer[fftSize - bufferSize], in,
               bufferSize * sizeof(float));
        // copy buffer since FFT is inplace
        std::copy(overlapSaveBuffer.begin(), overlapSaveBuffer.end(),
                  paddedIn.begin());

        fftConvolution(paddedIn.data(), filteredInput.data(), fftSize);

        // only use last bufferSize samples as output
        for (int i = 0; i < bufferSize; ++i)
        {
            out[i] = 0.3 * (filteredInput[fftSize - bufferSize + i] / fftSize);
        }

        // slide buffer
        memmove(overlapSaveBuffer.data(), &overlapSaveBuffer[bufferSize],
                (fftSize - bufferSize) * sizeof(float));
        break;
    case ConvolutionMethod::UniformOLS:
        uniformOLSProcess(in, out);
        break;
    default:
        break;
    }
}

void Convolver::uniformOLSProcess(const float *in, float *out)
{
    // copy new input to the tail of buffer
    memcpy(&overlapSaveBuffer[bufferSize], in, bufferSize * sizeof(float));
    // copy buffer since FFT is inplace
    std::copy(overlapSaveBuffer.begin(), overlapSaveBuffer.end(),
              paddedIn.begin());
    fftwf_plan p_fwd = fftwf_plan_dft_r2c_1d(
        2 * bufferSize, paddedIn.data(),
        reinterpret_cast<fftwf_complex *>(IN.data()), FFTW_ESTIMATE);
    fftwf_execute(p_fwd);
    fftwf_destroy_plan(p_fwd);

    // store input transform into FDL register
    std::copy(IN.begin(), IN.end(), fdlRegister.begin() + fdlRegisterPointer);
    for (int i = 0; i < numPartitions; ++i)
    {
        for (int j = 0; j < bufferSize + 1; ++j)
        {
            std::complex<float> fdl = fdlRegister[fdlRegisterPointer];
            std::complex<float> ir = partitionedIR[i * (bufferSize + 1) + j];
            float fdl_real = fdl.real();
            float fdl_imag = fdl.imag();
            float ir_real = ir.real();
            float ir_imag = ir.imag();
            Y[j] += std::complex<float>{
                fdl_real * ir_real - fdl_imag * ir_imag, // real
                fdl_real * ir_imag + fdl_imag * ir_real  // imag
            };
            // update pointer and check if it reached to the end
            if (++fdlRegisterPointer == fdlRegisterSize)
            {
                fdlRegisterPointer = 0;
            }
        }
    }

    fdlRegisterPointer =
        floorMod(fdlRegisterPointer - (bufferSize + 1), fdlRegister.size());
    fftwf_plan p_inv = fftwf_plan_dft_c2r_1d(
        2 * bufferSize, reinterpret_cast<fftwf_complex *>(Y.data()),
        filteredInput.data(), FFTW_ESTIMATE);
    fftwf_execute(p_inv);
    fftwf_destroy_plan(p_inv);
    // initialize Y
    std::fill(Y.begin(), Y.end(), std::complex<float>{0.0f, 0.0f});
    // only use last bufferSize samples as output
    for (int i = 0; i < bufferSize; ++i)
    {
        out[i] = 0.3 * (filteredInput[bufferSize + i] / (2 * bufferSize));
    }
    // slide buffer
    memmove(overlapSaveBuffer.data(), &overlapSaveBuffer[bufferSize],
            bufferSize * sizeof(float));
}

void Convolver::fftConvolution(float *in, float *out, std::size_t fftSize)
{
    fftwf_plan p_fwd = fftwf_plan_dft_r2c_1d(
        fftSize, in, reinterpret_cast<fftwf_complex *>(IN.data()),
        FFTW_ESTIMATE);
    fftwf_execute(p_fwd);
    fftwf_destroy_plan(p_fwd);

    for (int i = 0; i < fftSize / 2 + 1; ++i)
    {
        Y[i] = {
            IN[i].real() * IR[i].real() - IN[i].imag() * IR[i].imag(), // real
            IN[i].real() * IR[i].imag() + IN[i].imag() * IR[i].real()  // imag
        };
    }
    fftwf_plan p_inv = fftwf_plan_dft_c2r_1d(
        fftSize, reinterpret_cast<fftwf_complex *>(Y.data()),
        filteredInput.data(), FFTW_ESTIMATE);
    fftwf_execute(p_inv);
    fftwf_destroy_plan(p_inv);
    // only use last bufferSize samples as output
    for (int i = 0; i < bufferSize; ++i)
    {
        out[i] = 0.3 * (filteredInput[bufferSize + i] / fftSize);
    }

    // slide buffer
    memmove(overlapSaveBuffer.data(), &overlapSaveBuffer[bufferSize],
            (bufferSize) * sizeof(float));
}

void Convolver::setNewIR(std::complex<float> *newIR) { IR = newIR; }

void Convolver::setNewPartitionedIR(std::complex<float> *newIR)
{
    partitionedIR = newIR;
}