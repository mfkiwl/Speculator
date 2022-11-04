%module libstft
%{
#include <complex>
#include <vector>
#include "fft.h"
#include "stft.h"
%}

%inline %{
    struct FFTC2C
    {
        fft_complex * in, * out;
        fft_plan plan;    
        size_t size;

        FFTC2C(size_t n, int sign, unsigned int flags) {
            size = n;
            in = new fft_complex[n];
            out= new fft_complex[n];
            plan = fft_plan_dft_1d(n,in,out,sign,flags);
        }
        ~FFTC2C() {
            if(in) delete [] in;
            if(out) delete [] out;
            fft_destroy_plan(plan);
        }

        void execute(std::vector<std::complex<float>> & inputs, std::vector<std::complex<float>> & outputs)
        {
            for(size_t i = 0; i < size; i++)
            {
                in[i].real = inputs[i].real();
                in[i].imag = inputs[i].imag();
            }
            fft_execute(plan);
            for(size_t i= 0; i < size; i++)
            {
                outputs[i].real(out[i].real);
                outputs[i].imag(out[i].imag);
            }
        }
    };


    struct FFTR2C
    {
        fft_complex  * out;
        float * in;
        fft_plan plan;    
        size_t size;

        FFTR2C(size_t n, int sign, unsigned int flags) {
            size = n;
            in = new float[n];
            out= new fft_complex[n];
            plan = fft_plan_dft_r2c_1d(n,in,out,sign,flags);
        }
        ~FFTR2C() {
            if(in) delete [] in;
            if(out) delete [] out;
            fft_destroy_plan(plan);
        }

        void execute(std::vector<float>> & inputs, std::vector<std::complex<float>> & outputs)
        {
            memcpy(in,inputs.data(),n*sizeof(float));
            fft_execute(plan);
            for(size_t i= 0; i < size; i++)
            {
                outputs[i].real(out[i].real);
                outputs[i].imag(out[i].imag);
            }
        }
    };


    struct FFTC2R
    {
        fft_complex  * in;
        float * out;
        fft_plan plan;    
        size_t size;

        FFTC2R(size_t n, int sign, unsigned int flags) {
            size = n;
            out = new float[n];
            in new fft_complex[n];
            plan = fft_plan_dft_c2r_1d(n,in,out,sign,flags);
        }
        ~FFTC2R() {
            if(in) delete [] in;
            if(out) delete [] out;
            fft_destroy_plan(plan);
        }

        void execute(std::vector<std::complex<float>>> & inputs, std::vector<float> & outputs)
        {
            for(size_t i= 0; i < size; i++)
            {
                in[i].real = inputs[i].real();
                in[i].imag = inputs[i].imag();
            }            
            fft_execute(plan);
            memcpy(outputs.data(),out,n*sizeof(float));
            
        }
    };

    enum  WindowType
    {
        TRIANGULAR,
        HANN,
    };

        
    std::vector<std::complex<float>> run_stft(WindowType type, std::vector<float> & input, size_t data_size, size_t window_size, size_t hop_size) {
        size_t result_size = (data_size / hop_size);    
        std::vector<std::complex<float>> r(result_size * window_size);
        fft_complex * values = stft(input.data(),data_size, type == TRIANGULAR? triangular_window:hann_window,window_size,hop_size);
        assert(values != NULL);
        for(size_t i = 0; i < r.size(); i++)
        {
            r[i].real(values[i].real);
            r[i].imag(values[i].imag);
        }
        free(values);
    }

    std::vector<float> run_stft(WindowType type, std::vector<std::complex<float>> & input, size_t data_size, size_t window_size, size_t hop_size) {
        size_t result_size = data_size * hop_size + (window_size - hop_size);
        std::vector<float> r(result_size);            
        float * values = istft(input.data(),data_size, type == TRIANGULAR? triangular_window:hann_window,window_size,hop_size);
        assert(values != NULL);
        memcpy(r.data(),values,result_size * sizeof(float));
        free(values);
        return r;
    }
%}