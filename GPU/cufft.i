%module cufft 
%{

#include "cufft.h"

%}

%constant int cufft_forward = -1;
%constant int cufft_inverse = 1;

%include "std_vector.i"
%include "cufft.h"

typedef enum cufftType_t {
    CUFFT_R2C = 0x2a,  // Real to complex (interleaved) 
    CUFFT_C2R = 0x2c,  // Complex (interleaved) to real 
    CUFFT_C2C = 0x29,  // Complex to complex (interleaved) 
    CUFFT_D2Z = 0x6a,  // Double to double-complex (interleaved) 
    CUFFT_Z2D = 0x6c,  // Double-complex (interleaved) to double 
    CUFFT_Z2Z = 0x69   // Double-complex to double-complex (interleaved)
} cufftType;

typedef enum cudaDataType_t
{
        CUDA_R_16F= 2, // 16 bit real 
        CUDA_C_16F= 6, // 16 bit complex
        CUDA_R_32F= 0, // 32 bit real
        CUDA_C_32F= 4, // 32 bit complex
        CUDA_R_64F= 1, // 64 bit real
        CUDA_C_64F= 5, // 64 bit complex
        CUDA_R_8I= 3,  // 8 bit real as a signed integer 
        CUDA_C_8I= 7,  // 8 bit complex as a pair of signed integers
        CUDA_R_8U= 8,  // 8 bit real as a signed integer 
        CUDA_C_8U= 9   // 8 bit complex as a pair of signed integers
} cudaDataType;

typedef float               cufftReal;
typedef cuComplex           cufftComplex;

%template(float_vector)     std::vector<float>;
%template(complex_vector)   std::vector<cufftComplex>;

%inline %{

    cufftComplex make_complex(float r, float i) {
        return make_cuFloatComplex(r,i);
    }
 
    void print_complex(cufftComplex & x) {
        std::cout << "real=" << x.x << "imag=" << x.y << std::endl;
    }
    /*
    cufftComplex cabs(cufftComplex x) { return cabsf(x); }
    cufftComplex cacos(cufftComplex x) { return cacosf(x); }
    cufftComplex casin(cufftComplex x) { return casinf(x); }    
    cufftComplex catan(cufftComplex x) { return ccatanf(x); }
    cufftComplex ccos(cufftComplex x) { return cccosf(x); }
    cufftComplex csin(cufftComplex x) { return ccsinf(x); }    
    cufftComplex ctan(cufftComplex x) { return cctanf(x); }
    cufftComplex cacosh(cufftComplex x) { return cacoshf(x); }
    cufftComplex casinh(cufftComplex x) { return casinhf(x); }    
    cufftComplex catanh(cufftComplex x) { return catanhf(x); }
    cufftComplex ccosh(cufftComplex x) { return ccoshf(x); }
    cufftComplex csinh(cufftComplex x) { return csinhf(x); }    
    cufftComplex ctanh(cufftComplex x) { return ctanhf(x); }
    cufftComplex carg(cufftComplex x) { return cargf(x); }
    cufftComplex cconj(cufftComplex x) { return cconj(x); }
    cufftComplex cproj(cufftComplex x) { return cproj(x); }
    
    float creal(cufftComplex x) { return creal(x); }
    float cimag(cufftComplex x) { return cimag(x); }
    
    cufftComplex cexp(cufftComplex x) { return cexpf(x); }
    cufftComplex clog(cufftComplex x) { return clogf(x); }
    cufftComplex cpow(cufftComplex x, cufftComplex y) { return cpowf(x,y); }
    cufftComplex csqrt(cufftComplex x) { return csqrtf(x); }
    */

std::complex<float> to_complex(cufftComplex & x) {
    std::complex<float> r(x.x,x.y);
    return r;
}


%}