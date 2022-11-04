#ifndef ANALYSIS_FFT_H
#define ANALYSIS_FFT_H

#include "rpcxx.h"
#include <fftw3.h>
#include <complex>
#include <memory>
#include <QMutex>

#if defined(EMSCRIPTEN)
#   define FFTW_EM_FLAG FFTW_ESTIMATE
#else
#   define FFTW_EM_FLAG FFTW_MEASURE
#endif

namespace std
{
    using dcomplex = complex<double>;

    inline dcomplex& cast_dcomplex(fftw_complex& z)
    {
        return reinterpret_cast<dcomplex&>(z);
    }

    inline const dcomplex& cast_dcomplex(const fftw_complex& z)
    {
        return reinterpret_cast<const dcomplex&>(z);
    }
}

namespace Analysis
{
    extern QMutex sFFTWPlanMutex;

    void importFFTWisdom();

    class ReReFFT
    {
    public:
        ReReFFT(int n, fftw_r2r_kind method);
        ~ReReFFT();
        
        double data(int index) const;
        double& data(int index);

        void compute();

        int getLength() const;

    private:
        void checkIndex(int index) const;

        int mSize;
        fftw_plan mPlan;

        double *mData;
    };

    class RealFFT
    {
    public:
        RealFFT(int n);
        ~RealFFT();

        double input(int index) const;
        double& input(int index);

        std::dcomplex output(int index) const;
        std::dcomplex& output(int index);

        void computeForward();
        void computeBackward();

        int getInputLength() const;
        int getOutputLength() const;

    private:
        void checkInputIndex(int index) const;
        void checkOutputIndex(int index) const;

        int mSize;
        fftw_plan mPlanForward;
        fftw_plan mPlanBackward;

        double *mIn;
        fftw_complex *mOut;
    };

    class ComplexFFT
    {
    public:
        ComplexFFT(int n);
        ~ComplexFFT();

        std::dcomplex data(int index) const;
        std::dcomplex& data(int index);
        
        void computeForward();
        void computeBackward();

        int getLength() const;

    private:
        void checkIndex(int index) const;

        int mSize;
        fftw_plan mPlanForward;
        fftw_plan mPlanBackward;

        fftw_complex *mData;
    };

    rpm::vector<double> fft_n(Analysis::RealFFT *fft, const rpm::vector<double>& signal, rpm::map<int, rpm::vector<double>>& windowCache);
}

#endif // ANALYSIS_FFT_H
