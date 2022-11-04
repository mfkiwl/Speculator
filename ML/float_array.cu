#include <cassert>
#include "cuda_runtime.h"
#include "math_constants.h"
#include <map>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "vector_float.h"
#include "matrix_float.h"
#include "cube_float.h"
#include "field_float.h"

int dims(int M, int N, int O, int P) {
    int n = 1;
    if(N > 0) n++;
    if(O > 0) n++;
    if(P > 0) n++;
    return n;
}
int size(int M, int N, int O, int P) {
    return M*(N>0? N:1)*(O>0?O:1)*(P>0?P:1);    
}

float* array_addf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_addf(x,y,M);
    if(d == 2) return _2d_addf(x,y,M,N);
    if(d == 3) return cube_addf(x,y,M,N,O);
    if(d == 4) return field_addf(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_addf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_addf(x,y,output,M);
    else if(d == 2) _2d_r_addf(x,y,output,M,N);
    else if(d == 3) cube_r_addf(x,y,output,M,N,O);
    else if(d == 4) field_r_addf(x,y,output, M,N,O,P);
       
}

float* array_subf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_subf(x,y,M);
    if(d == 2) return _2d_subf(x,y,M,N);
    if(d == 3) return cube_subf(x,y,M,N,O);
    if(d == 4) return field_subf(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_subf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_subf(x,y,output,M);
    else if(d == 2) _2d_r_subf(x,y,output,M,N);
    else if(d == 3) cube_r_subf(x,y,output,M,N,O);
    else if(d == 4) field_r_subf(x,y,output, M,N,O,P);
       
}

float* array_mulf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_mulf(x,y,M);
    if(d == 2) return _2d_mulf(x,y,M,N);
    if(d == 3) return cube_mulf(x,y,M,N,O);
    if(d == 4) return field_mulf(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_mulf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_mulf(x,y,output,M);
    else if(d == 2) _2d_r_mulf(x,y,output,M,N);
    else if(d == 3) cube_r_mulf(x,y,output,M,N,O);
    else if(d == 4) field_r_mulf(x,y,output, M,N,O,P);
       

}


float* array_divf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_divf(x,y,M);
    if(d == 2) return _2d_divf(x,y,M,N);
    if(d == 3) return cube_divf(x,y,M,N,O);
    if(d == 4) return field_divf(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_divf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_divf(x,y,output,M);
    else if(d == 2) _2d_r_divf(x,y,output,M,N);
    else if(d == 3) cube_r_divf(x,y,output,M,N,O);
    else if(d == 4) field_r_divf(x,y,output, M,N,O,P);
       
}

float* array_modf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_modf(x,y,M);
    if(d == 2) return _2d_modf(x,y,M,N);
    if(d == 3) return cube_modf(x,y,M,N,O);
    if(d == 4) return field_modf(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_modf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_modf(x,y,output,M);
    else if(d == 2) _2d_r_modf(x,y,output,M,N);
    else if(d == 3) cube_r_modf(x,y,output,M,N,O);
    else if(d == 4) field_r_modf(x,y,output, M,N,O,P);
       
}

float* array_acosf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_acosf(x,M);
    if(d == 2) return matrix_acosf(x,M,N);
    if(d == 3) return cube_acosf(x,M,N,O);
    if(d == 4) return field_acosf(x,M,N,O,P);
    
    return NULL;
}

void array_r_acosf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_acosf(x,output,M);
    else if(d == 2) matrix_r_acosf(x,output,M,N);
    else if(d == 3) cube_r_acosf(x,output,M,N,O);
    else if(d == 4) field_r_acosf(x,output, M,N,O,P);
       
}

float* array_acoshf(float * x, int M,int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_acoshf(x,M);
    if(d == 2) return matrix_acoshf(x,M,N);
    if(d == 3) return cube_acoshf(x,M,N,O);
    if(d == 4) return field_acoshf(x,M,N,O,P);
    
    return NULL;
}

void array_r_acoshf(float * x, float * output, int M,int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_acoshf(x,output,M);
    else if(d == 2) matrix_r_acoshf(x,output,M,N);
    else if(d == 3) cube_r_acoshf(x,output,M,N,O);
    else if(d == 4) field_r_acoshf(x,output, M,N,O,P);
       
}


float* array_asinhf(float * x, int M,int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_asinhf(x,M);
    if(d == 2) return matrix_asinhf(x,M,N);
    if(d == 3) return cube_asinhf(x,M,N,O);
    if(d == 4) return field_asinhf(x,M,N,O,P);
    
    return NULL;
}

void array_r_asinhf(float *x, float * output, int M,int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_asinhf(x,output,M);
    else if(d == 2) matrix_r_asinhf(x,output,M,N);
    else if(d == 3) cube_r_asinhf(x,output,M,N,O);
    else if(d == 4) field_r_asinhf(x,output, M,N,O,P);
       
}

float* array_asinf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_asinf(x,M);
    if(d == 2) return matrix_asinf(x,M,N);
    if(d == 3) return cube_asinf(x,M,N,O);
    if(d == 4) return field_asinf(x,M,N,O,P);
    
    return NULL;
}

void array_r_asinf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_asinf(x,output,M);
    else if(d == 2) matrix_r_asinf(x,output,M,N);
    else if(d == 3) cube_r_asinf(x,output,M,N,O);
    else if(d == 4) field_r_asinf(x,output, M,N,O,P);
       
}


float* array_atan2f(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_atan2f(x,y,M);
    if(d == 2) return matrix_atan2f(x,y,M,N);
    if(d == 3) return cube_atan2f(x,y,M,N,O);
    if(d == 4) return field_atan2f(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_atan2f(float * x, float * y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_atan2f(x,y,output,M);
    else if(d == 2) matrix_r_atan2f(x,y,output,M,N);
    else if(d == 3) cube_r_atan2f(x,y,output,M,N,O);
    else if(d == 4) field_r_atan2f(x,y,output, M,N,O,P);
       
}

float* array_atanf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_atanf(x,M);
    if(d == 2) return matrix_atanf(x,M,N);
    if(d == 3) return cube_atanf(x,M,N,O);
    if(d == 4) return field_atanf(x,M,N,O,P);
    
    return NULL;
}

void array_r_atanf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_atanf(x,output,M);
    else if(d == 2) matrix_r_atanf(x,output,M,N);
    else if(d == 3) cube_r_atanf(x,output,M,N,O);
    else if(d == 4) field_r_atanf(x,output, M,N,O,P);
           
}

float* array_atanhf(float * x, int M,int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_atanhf(x,M);
    if(d == 2) return matrix_atanhf(x,M,N);
    if(d == 3) return cube_atanhf(x,M,N,O);
    if(d == 4) return field_atanhf(x,M,N,O,P);
    
    return NULL;
}

void array_r_atanhf(float * x, float * output, int M,int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_atanhf(x,output,M);
    else if(d == 2) matrix_r_atanhf(x,output,M,N);
    else if(d == 3) cube_r_atanhf(x,output,M,N,O);
    else if(d == 4) field_r_atanhf(x,output, M,N,O,P);
           
}

float* array_ceilf(float * x, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_ceilf(x,M);
    if(d == 2) return matrix_ceilf(x,M,N);
    if(d == 3) return cube_ceilf(x,M,N,O);
    if(d == 4) return field_ceilf(x,M,N,O,P);
    
    return NULL;
}

void array_r_ceilf(float * x, float * output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_ceilf(x,output,M);
    else if(d == 2) matrix_r_ceilf(x,output,M,N);
    else if(d == 3) cube_r_ceilf(x,output,M,N,O);
    else if(d == 4) field_r_ceilf(x,output, M,N,O,P);
           
}

float* array_cosf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_cosf(x,M);
    if(d == 2) return matrix_cosf(x,M,N);
    if(d == 3) return cube_cosf(x,M,N,O);
    if(d == 4) return field_cosf(x,M,N,O,P);
    
    return NULL;
}
void array_r_cosf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_acosf(x,output,M);
    else if(d == 2) matrix_r_acosf(x,output,M,N);
    else if(d == 3) cube_r_acosf(x,output,M,N,O);
    else if(d == 4) field_r_acosf(x,output, M,N,O,P);
           
}


float* array_coshf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_coshf(x,M);
    if(d == 2) return matrix_coshf(x,M,N);
    if(d == 3) return cube_coshf(x,M,N,O);
    if(d == 4) return field_coshf(x,M,N,O,P);
    
    return NULL;
}
void array_r_coshf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_acoshf(x,output,M);
    else if(d == 2) matrix_r_coshf(x,output,M,N);
    else if(d == 3) cube_r_coshf(x,output,M,N,O);
    else if(d == 4) field_r_coshf(x,output, M,N,O,P);
           
}

float* array_exp10f(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_exp10f(x,M);
    if(d == 2) return matrix_exp10f(x,M,N);
    if(d == 3) return cube_exp10f(x,M,N,O);
    if(d == 4) return field_exp10f(x,M,N,O,P);
    
    return NULL;
}
void array_r_exp10f(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_exp10f(x,output,M);
    else if(d == 2) matrix_r_exp10f(x,output,M,N);
    else if(d == 3) cube_r_exp10f(x,output,M,N,O);
    else if(d == 4) field_r_exp10f(x,output, M,N,O,P);
           
}

float* array_exp2f(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_exp2f(x,M);
    if(d == 2) return matrix_exp2f(x,M,N);
    if(d == 3) return cube_exp2f(x,M,N,O);
    if(d == 4) return field_exp2f(x,M,N,O,P);
    
    return NULL;
}
void array_r_exp2f(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_exp2f(x,output,M);
    else if(d == 2) matrix_r_exp2f(x,output,M,N);
    else if(d == 3) cube_r_exp2f(x,output,M,N,O);
    else if(d == 4) field_r_exp2f(x,output, M,N,O,P);
           
}

float* array_expf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);    
    if(d == 1) return vector_expf(x,M);
    if(d == 2) return matrix_expf(x,M,N);
    if(d == 3) return cube_expf(x,M,N,O);
    if(d == 4) return field_expf(x,M,N,O,P);
    
    return NULL;
}
void array_r_expf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_expf(x,output,M);
    else if(d == 2) matrix_r_expf(x,output,M,N);
    else if(d == 3) cube_r_expf(x,output,M,N,O);
    else if(d == 4) field_r_expf(x,output, M,N,O,P);
           
}

float* array_expm1f(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_expm1f(x,M);
    if(d == 2) return matrix_expm1f(x,M,N);
    if(d == 3) return cube_expm1f(x,M,N,O);
    if(d == 4) return field_expm1f(x,M,N,O,P);
    
    return NULL;
}
void array_r_expm1f(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_expm1f(x,output,M);
    else if(d == 2) matrix_r_expm1f(x,output,M,N);
    else if(d == 3) cube_r_expm1f(x,output,M,N,O);
    else if(d == 4) field_r_expm1f(x,output, M,N,O,P);
           
}

float* array_fabsf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fabsf(x,M);
    if(d == 2) return matrix_fabsf(x,M,N);
    if(d == 3) return cube_fabsf(x,M,N,O);
    if(d == 4) return field_fabsf(x,M,N,O,P);
    
    return NULL;
}
void array_r_fabsf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fabsf(x,output,M);
    else if(d == 2) matrix_r_fabsf(x,output,M,N);
    else if(d == 3) cube_r_fabsf(x,output,M,N,O);
    else if(d == 4) field_r_fabsf(x,output, M,N,O,P);
           
}

float* array_floorf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_floorf(x,M);
    if(d == 2) return matrix_floorf(x,M,N);
    if(d == 3) return cube_floorf(x,M,N,O);
    if(d == 4) return field_floorf(x,M,N,O,P);
    
    return NULL;
}
void array_r_floorf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_floorf(x,output,M);
    else if(d == 2) matrix_r_floorf(x,output,M,N);
    else if(d == 3) cube_r_floorf(x,output,M,N,O);
    else if(d == 4) field_r_floorf(x,output, M,N,O,P);
           
}

float* array_fmaxf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fmaxf(x,y,M);
    if(d == 2) return matrix_fmaxf(x,y,M,N);
    if(d == 3) return cube_fmaxf(x,y,M,N,O);
    if(d == 4) return field_fmaxf(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_fmaxf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fmaxf(x,y,output,M);
    else if(d == 2) matrix_r_fmaxf(x,y,output,M,N);
    else if(d == 3) cube_r_fmaxf(x,y,output,M,N,O);
    else if(d == 4) field_r_fmaxf(x,y,output, M,N,O,P);
           
}

float* array_fminf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fminf(x,y,M);
    if(d == 2) return matrix_fminf(x,y,M,N);
    if(d == 3) return cube_fminf(x,y,M,N,O);
    if(d == 4) return field_fminf(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_fminf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fminf(x,y,output,M);
    else if(d == 2) matrix_r_fminf(x,y,output,M,N);
    else if(d == 3) cube_r_fminf(x,y,output,M,N,O);
    else if(d == 4) field_r_fminf(x,y,output, M,N,O,P);
           
}

float* array_fmodf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fmodf(x,y,M);
    if(d == 2) return matrix_fmodf(x,y,M,N);
    if(d == 3) return cube_fmodf(x,y,M,N,O);
    if(d == 4) return field_fmodf(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_fmodf(float * x, float * y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fmodf(x,y,output,M);
    else if(d == 2) matrix_r_fmodf(x,y,output,M,N);
    else if(d == 3) cube_r_fmodf(x,y,output,M,N,O);
    else if(d == 4) field_r_fmodf(x,y,output, M,N,O,P);
           
}

float* array_log10f(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_log10f(x,M);
    if(d == 2) return matrix_log10f(x,M,N);
    if(d == 3) return cube_log10f(x,M,N,O);
    if(d == 4) return field_log10f(x,M,N,O,P);
    
    return NULL;
}
void array_r_log10f(float * x,float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_log10f(x,output,M);
    else if(d == 2) matrix_r_log10f(x,output,M,N);
    else if(d == 3) cube_r_log10f(x,output,M,N,O);
    else if(d == 4) field_r_log10f(x,output, M,N,O,P);
           
}

float* array_log1pf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_log1pf(x,M);
    if(d == 2) return matrix_log1pf(x,M,N);
    if(d == 3) return cube_log1pf(x,M,N,O);
    if(d == 4) return field_log1pf(x,M,N,O,P);
    
    return NULL;
}
void array_r_log1pf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_log1pf(x,output,M);
    else if(d == 2) matrix_r_log1pf(x,output,M,N);
    else if(d == 3) cube_r_log1pf(x,output,M,N,O);
    else if(d == 4) field_r_log1pf(x,output, M,N,O,P);
           
}

float* array_log2f(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_log2f(x,M);
    if(d == 2) return matrix_log2f(x,M,N);
    if(d == 3) return cube_log2f(x,M,N,O);
    if(d == 4) return field_log2f(x,M,N,O,P);
    
    return NULL;
}

void array_r_log2f(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_log2f(x,output,M);
    else if(d == 2) matrix_r_log2f(x,output,M,N);
    else if(d == 3) cube_r_log2f(x,output,M,N,O);
    else if(d == 4) field_r_log2f(x,output, M,N,O,P);
           
}

float* array_logbf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_logbf(x,M);
    if(d == 2) return matrix_logbf(x,M,N);
    if(d == 3) return cube_logbf(x,M,N,O);
    if(d == 4) return field_logbf(x,M,N,O,P);
    
    return NULL;
}
void array_r_logbf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_logbf(x,output,M);
    else if(d == 2) matrix_r_logbf(x,output,M,N);
    else if(d == 3) cube_r_logbf(x,output,M,N,O);
    else if(d == 4) field_r_logbf(x,output, M,N,O,P);
           
}

float* array_powf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_powf(x,y,M);
    if(d == 2) return matrix_powf(x,y,M,N);
    if(d == 3) return cube_powf(x,y,M,N,O);
    if(d == 4) return field_powf(x,y,M,N,O,P);
    
    return NULL;
}    

void array_r_powf(float * x, float * y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_powf(x,y,output,M);
    else if(d == 2) matrix_r_powf(x,y,output,M,N);
    else if(d == 3) cube_r_powf(x,y,output,M,N,O);
    else if(d == 4) field_r_powf(x,y,output, M,N,O,P);
       
}

float* array_rsqrtf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_rsqrtf(x,M);
    if(d == 2) return matrix_rsqrtf(x,M,N);
    if(d == 3) return cube_rsqrtf(x,M,N,O);
    if(d == 4) return field_rsqrtf(x,M,N,O,P);
    
    return NULL;
}

void array_r_rsqrtf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_rsqrtf(x,output,M);
    else if(d == 2) matrix_r_rsqrtf(x,output,M,N);
    else if(d == 3) cube_r_rsqrtf(x,output,M,N,O);
    else if(d == 4) field_r_rsqrtf(x,output, M,N,O,P);
           
}

float* array_sinf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_sinf(x,M);
    if(d == 2) return matrix_sinf(x,M,N);
    if(d == 3) return cube_sinf(x,M,N,O);
    if(d == 4) return field_sinf(x,M,N,O,P);
    
    return NULL;
}
void array_r_sinf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_sinf(x,output,M);
    else if(d == 2) matrix_r_sinf(x,output,M,N);
    else if(d == 3) cube_r_sinf(x,output,M,N,O);
    else if(d == 4) field_r_sinf(x,output, M,N,O,P);
           
}

float* array_sinhf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_sinhf(x,M);
    if(d == 2) return matrix_sinhf(x,M,N);
    if(d == 3) return cube_sinhf(x,M,N,O);
    if(d == 4) return field_sinhf(x,M,N,O,P);
    
    return NULL;
}

float* array_r_sinhf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_sinhf(x,output,M);
    else if(d == 2) matrix_r_sinhf(x,output,M,N);
    else if(d == 3) cube_r_sinhf(x,output,M,N,O);
    else if(d == 4) field_r_sinhf(x,output, M,N,O,P);
           
    return NULL;
}

float* array_sqrtf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_rsqrtf(x,M);
    if(d == 2) return matrix_rsqrtf(x,M,N);
    if(d == 3) return cube_rsqrtf(x,M,N,O);
    if(d == 4) return field_rsqrtf(x,M,N,O,P);
    
    return NULL;
}

void array_r_sqrtf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_sqrtf(x,output,M);
    else if(d == 2) matrix_r_sqrtf(x,output,M,N);
    else if(d == 3) cube_r_sqrtf(x,output,M,N,O);
    else if(d == 4) field_r_sqrtf(x,output, M,N,O,P);
           
}

float* array_tanf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_tanf(x,M);
    if(d == 2) return matrix_tanf(x,M,N);
    if(d == 3) return cube_tanf(x,M,N,O);
    if(d == 4) return field_tanf(x,M,N,O,P);
    
    return NULL;
}
float* array_r_tanf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_tanf(x,output,M);
    else if(d == 2) matrix_r_tanf(x,output,M,N);
    else if(d == 3) cube_r_tanf(x,output,M,N,O);
    else if(d == 4) field_r_tanf(x,output, M,N,O,P);
           
    return NULL;
}

float* array_tanhf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);    
    if(d == 1) return vector_tanhf(x,M);
    if(d == 2) return matrix_tanhf(x,M,N);
    if(d == 3) return cube_tanhf(x,M,N,O);
    if(d == 4) return field_tanhf(x,M,N,O,P);
    
    return NULL;
}
void array_r_tanhf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_tanhf(x,output,M);
    else if(d == 2) matrix_r_tanhf(x,output,M,N);
    else if(d == 3) cube_r_tanhf(x,output,M,N,O);
    else if(d == 4) field_r_tanhf(x,output, M,N,O,P);
           
}

float* array_softmaxf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_softmaxf(x,M);
    if(d == 2) return matrix_softmaxf(x,M,N);
    if(d == 3) return cube_softmaxf(x,M,N,O);
    if(d == 4) return field_softmaxf(x,M,N,O,P);
    
    return NULL;
}
void array_r_softmaxf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_softmaxf(x,output,M);
    else if(d == 2) matrix_r_softmaxf(x,output,M,N);
    else if(d == 3) cube_r_softmaxf(x,output,M,N,O);
    else if(d == 4) field_r_softmaxf(x,output, M,N,O,P);
           
}

float* array_sigmoidf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);    
    if(d == 1) return vector_sigmoidf(x,M);
    if(d == 2) return matrix_sigmoidf(x,M,N);
    if(d == 3) return cube_sigmoidf(x,M,N,O);
    if(d == 4) return field_sigmoidf(x,M,N,O,P);
    
    return NULL;
}
void array_r_sigmoidf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_sigmoidf(x,output,M);
    else if(d == 2) matrix_r_sigmoidf(x,output,M,N);
    else if(d == 3) cube_r_sigmoidf(x,output,M,N,O);
    else if(d == 4) field_r_sigmoidf(x,output, M,N,O,P);
           
}
float* array_sigmoid_gradf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);    
    if(d == 1) return vector_sigmoid_gradf(x,M);
    if(d == 2) return matrix_sigmoid_gradf(x,M,N);
    if(d == 3) return cube_sigmoid_gradf(x,M,N,O);
    if(d == 4) return field_sigmoid_gradf(x,M,N,O,P);    
    return NULL;
}
void array_r_sigmoid_gradf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_sigmoid_gradf(x,output,M);
    else if(d == 2) matrix_r_sigmoid_gradf(x,output,M,N);
    else if(d == 3) cube_r_sigmoid_gradf(x,output,M,N,O);
    else if(d == 4) field_r_sigmoid_gradf(x,output, M,N,O,P);
           
}

float* array_tanh_gradf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_tanh_gradf(x,M);
    if(d == 2) return matrix_tanh_gradf(x,M,N);
    if(d == 3) return cube_tanh_gradf(x,M,N,O);
    if(d == 4) return field_tanh_gradf(x,M,N,O,P);
    
    return NULL;
}

void array_r_tanh_gradf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_tanh_gradf(x,output,M);
    if(d == 2) matrix_r_tanh_gradf(x,output,M,N);
    if(d == 3) cube_r_tanh_gradf(x,output,M,N,O);
    if(d == 4) field_r_tanh_gradf(x,output,M,N,O,P);
        
}

float* array_reluf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_reluf(x,M);
    if(d == 2) return matrix_reluf(x,M,N);
    if(d == 3) return cube_reluf(x,M,N,O);
    if(d == 4) return field_reluf(x,M,N,O,P);
    
    return NULL;
}
void array_r_reluf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_reluf(x,output,M);
    if(d == 2) matrix_r_reluf(x,output,M,N);
    if(d == 3) cube_r_reluf(x,output,M,N,O);
    if(d == 4) field_r_reluf(x,output,M,N,O,P);
        
}

float* array_relu_gradf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_relu_gradf(x,M);
    if(d == 2) return matrix_relu_gradf(x,M,N);
    if(d == 3) return cube_relu_gradf(x,M,N,O);
    if(d == 4) return field_relu_gradf(x,M,N,O,P);
    
    return NULL;
}
void array_r_relu_gradf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_relu_gradf(x,output,M);
    if(d == 2) matrix_r_relu_gradf(x,output,M,N);
    if(d == 3) cube_r_relu_gradf(x,output,M,N,O);
    if(d == 4) field_r_relu_gradf(x,output,M,N,O,P);
        
}

float* array_addf_const(float * x, float y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_addf_const(x,y,M);
    if(d == 2) return matrix_addf_const(x,y,M,N);
    if(d == 3) return cube_addf_const(x,y,M,N,O);
    if(d == 4) return field_addf_const(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_addf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_addf_const(x,y,output,M);
    if(d == 2) matrix_r_addf_const(x,y,output,M,N);
    if(d == 3) cube_r_addf_const(x,y,output,M,N,O);
    if(d == 4) field_r_addf_const(x,y,output,M,N,O,P);
        
}

float* array_subf_const(float * x, float y, int M, int N, int O, int P)
{  
    int d = dims(M,N,O,P);
    if(d == 1) return vector_subf_const(x,y,M);
    if(d == 2) return matrix_subf_const(x,y,M,N);
    if(d == 3) return cube_subf_const(x,y,M,N,O);
    if(d == 4) return field_subf_const(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_subf_const(float * x, float y, float *output,  int M, int N, int O, int P)
{  
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_subf_const(x,y,output,M);
    if(d == 2) matrix_r_subf_const(x,y,output,M,N);
    if(d == 3) cube_r_subf_const(x,y,output,M,N,O);
    if(d == 4) field_r_subf_const(x,y,output,M,N,O,P);
        
}

float* array_mulf_const(float * x, float y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_mulf_const(x,y,M);
    if(d == 2) return matrix_mulf_const(x,y,M,N);
    if(d == 3) return cube_mulf_const(x,y,M,N,O);
    if(d == 4) return field_mulf_const(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_mulf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);    
    if(d == 1) vector_r_mulf_const(x,y,output,M);
    if(d == 2) matrix_r_mulf_const(x,y,output,M,N);
    if(d == 3) cube_r_mulf_const(x,y,output,M,N,O);
    if(d == 4) field_r_mulf_const(x,y,output,M,N,O,P);       
}

float* array_divf_const(float * x, float y, int M, int O, int N, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_divf_const(x,y,M);
    if(d == 2) return matrix_divf_const(x,y,M,N);
    if(d == 3) return cube_divf_const(x,y,M,N,O);
    if(d == 4) return field_divf_const(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_divf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_divf_const(x,y,output,M);
    if(d == 2) matrix_r_divf_const(x,y,output,M,N);
    if(d == 3) cube_r_divf_const(x,y,output,M,N,O);
    if(d == 4) field_r_divf_const(x,y,output,M,N,O,P);
        
}


float* array_modf_const(float * x, float y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_modf_const(x,y,M);
    if(d == 2) return matrix_modf_const(x,y,M,N);
    if(d == 3) return cube_modf_const(x,y,M,N,O);
    if(d == 4) return field_modf_const(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_modf_const(float * x, float y, float * output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_modf_const(x,y,output,M);
    if(d == 2) matrix_r_modf_const(x,y,output,M,N);
    if(d == 3) cube_r_modf_const(x,y,output,M,N,O);
    if(d == 4) field_r_modf_const(x,y,output,M,N,O,P);
    
}

float* array_atan2f_const(float * x, float  y, int M,int N, int O, int P)
{   
    int d = dims(M,N,O,P);
    if(d == 1) return vector_atan2f_const(x,y,M);
    if(d == 2) return matrix_atan2f_const(x,y,M,N);
    if(d == 3) return cube_atan2f_const(x,y,M,N,O);
    if(d == 4) return field_atan2f_const(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_atan2f_const(float * x, float  y, float *output, int M,int N, int O, int P)
{   
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_atan2f_const(x,y,output,M);
    if(d == 2) matrix_r_atan2f_const(x,y,output,M,N);
    if(d == 3) cube_r_atan2f_const(x,y,output,M,N,O);
    if(d == 4) field_r_atan2f_const(x,y,output,M,N,O,P);
    
}

float* array_fmaxf_const(float * x, float y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fmaxf_const(x,y,M);
    if(d == 2) return matrix_fmaxf_const(x,y,M,N);
    if(d == 3) return cube_fmaxf_const(x,y,M,N,O);
    if(d == 4) return field_fmaxf_const(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_fmaxf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fmaxf_const(x,y,output,M);
    if(d == 2) matrix_r_fmaxf_const(x,y,output,M,N);
    if(d == 3) cube_r_fmaxf_const(x,y,output,M,N,O);
    if(d == 4) field_r_fmaxf_const(x,y,output,M,N,O,P);
        
}

float* array_fminf_const(float * x, float y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fminf_const(x,y,M);
    if(d == 2) return matrix_fminf_const(x,y,M,N);
    if(d == 3) return cube_fminf_const(x,y,M,N,O);
    if(d == 4) return field_fminf_const(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_fminf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fminf_const(x,y,output,M);
    if(d == 2) matrix_r_fminf_const(x,y,output,M,N);
    if(d == 3) cube_r_fminf_const(x,y,output,M,N,O);
    if(d == 4) field_r_fminf_const(x,y,output,M,N,O,P);
        
}

float* array_fmodf_const(float * x, float y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fmodf_const(x,y,M);
    if(d == 2) return matrix_fmodf_const(x,y,M,N);
    if(d == 3) return cube_fmodf_const(x,y,M,N,O);
    if(d == 4) return field_fmodf_const(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_fmodf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fmodf_const(x,y,output,M);
    if(d == 2) matrix_r_fmodf_const(x,y,output,M,N);
    if(d == 3) cube_r_fmodf_const(x,y,output,M,N,O);
    if(d == 4) field_r_fmodf_const(x,y,output,M,N,O,P);
        
}

float* array_powf_const(float * x, float y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_powf_const(x,y,M);
    if(d == 2) return matrix_powf_const(x,y,M,N);
    if(d == 3) return cube_powf_const(x,y,M,N,O);
    if(d == 4) return field_powf_const(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_powf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_powf_const(x,y,output,M);
    if(d == 2) matrix_r_powf_const(x,y,output,M,N);
    if(d == 3) cube_r_powf_const(x,y,output,M,N,O);
    if(d == 4) field_r_powf_const(x,y,output,M,N,O,P);
        
}

float* array_addf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_addf_scalar(x,y,M);
    if(d == 2) return matrix_addf_scalar(x,y,M,N);
    if(d == 3) return cube_addf_scalar(x,y,M,N,O);
    if(d == 4) return field_addf_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_addf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_addf_scalar(x,y,output,M);
    if(d == 2) matrix_r_addf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_addf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_addf_scalar(x,y,output,M,N,O,P);
        
}

float* array_subf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_subf_scalar(x,y,M);
    if(d == 2) return matrix_subf_scalar(x,y,M,N);
    if(d == 3) return cube_subf_scalar(x,y,M,N,O);
    if(d == 4) return field_subf_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_subf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_subf_scalar(x,y,output,M);
    if(d == 2) matrix_r_subf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_subf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_subf_scalar(x,y,output,M,N,O,P);
        
}

float* array_mulf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_mulf_scalar(x,y,M);
    if(d == 2) return matrix_mulf_scalar(x,y,M,N);
    if(d == 3) return cube_mulf_scalar(x,y,M,N,O);
    if(d == 4) return field_mulf_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_mulf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_mulf_scalar(x,y,output,M);
    if(d == 2) matrix_r_mulf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_mulf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_mulf_scalar(x,y,output,M,N,O,P);
    
}

float* array_divf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_divf_scalar(x,y,M);
    if(d == 2) return matrix_divf_scalar(x,y,M,N);
    if(d == 3) return cube_divf_scalar(x,y,M,N,O);
    if(d == 4) return field_divf_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_divf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_divf_scalar(x,y,output,M);
    if(d == 2) matrix_r_divf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_divf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_divf_scalar(x,y,output,M,N,O,P);
        
}

float* array_modf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_modf_scalar(x,y,M);
    if(d == 2) return matrix_modf_scalar(x,y,M,N);
    if(d == 3) return cube_modf_scalar(x,y,M,N,O);
    if(d == 4) return field_modf_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_modf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_modf_scalar(x,y,output,M);
    if(d == 2) matrix_r_modf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_modf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_modf_scalar(x,y,output,M,N,O,P);
        
}
float* array_fmaxf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fmaxf_scalar(x,y,M);
    if(d == 2) return matrix_fmaxf_scalar(x,y,M,N);
    if(d == 3) return cube_fmaxf_scalar(x,y,M,N,O);
    if(d == 4) return field_fmaxf_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_fmaxf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fmaxf_scalar(x,y,output,M);
    if(d == 2) matrix_r_fmaxf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_fmaxf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_fmaxf_scalar(x,y,output,M,N,O,P);
        
}

float* array_fminf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fminf_scalar(x,y,M);
    if(d == 2) return matrix_fminf_scalar(x,y,M,N);
    if(d == 3) return cube_fminf_scalar(x,y,M,N,O);
    if(d == 4) return field_fminf_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_fminf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fminf_scalar(x,y,output,M);
    if(d == 2) matrix_r_fminf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_fminf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_fminf_scalar(x,y,output,M,N,O,P);
        
}

float* array_powf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_powf_scalar(x,y,M);
    if(d == 2) return matrix_powf_scalar(x,y,M,N);
    if(d == 3) return cube_powf_scalar(x,y,M,N,O);
    if(d == 4) return field_powf_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_powf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_powf_scalar(x,y,output,M);
    if(d == 2) matrix_r_powf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_powf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_powf_scalar(x,y,output,M,N,O,P);
        
}

float* array_hypotf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_hypotf_scalar(x,y,M);
    if(d == 2) return matrix_hypotf_scalar(x,y,M,N);
    if(d == 3) return cube_hypotf_scalar(x,y,M,N,O);
    if(d == 4) return field_hypotf_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_hypotf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_hypotf_scalar(x,y,output,M);
    if(d == 2) matrix_r_hypotf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_hypotf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_hypotf_scalar(x,y,output,M,N,O,P);
        
}

;

float* array_rhypotf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_rhypotf_scalar(x,y,M);
    if(d == 2) return matrix_rhypotf_scalar(x,y,M,N);
    if(d == 3) return cube_rhypotf_scalar(x,y,M,N,O);
    if(d == 4) return field_rhypotf_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_rhypotf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_rhypotf_scalar(x,y,output,M);
    if(d == 2) matrix_r_rhypotf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_rhypotf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_rhypotf_scalar(x,y,output,M,N,O,P);
        
}

float* array_fdividef_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fdividef_scalar(x,y,M);
    if(d == 2) return matrix_fdividef_scalar(x,y,M,N);
    if(d == 3) return cube_fdividef_scalar(x,y,M,N,O);
    if(d == 4) return field_fdividef_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_fdividef_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fdividef_scalar(x,y,output,M);
    if(d == 2) matrix_r_fdividef_scalar(x,y,output,M,N);
    if(d == 3) cube_r_fdividef_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_fdividef_scalar(x,y,output,M,N,O,P);
        
}
float* array_fdimf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fdimf_scalar(x,y,M);
    if(d == 2) return matrix_fdimf_scalar(x,y,M,N);
    if(d == 3) return cube_fdimf_scalar(x,y,M,N,O);
    if(d == 4) return field_fdimf_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_fdimf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fdimf_scalar(x,y,output,M);
    if(d == 2) matrix_r_fdimf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_fdimf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_fdimf_scalar(x,y,output,M,N,O,P);
        
}


float* array_fmodf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fmodf_scalar(x,y,M);
    if(d == 2) return matrix_fmodf_scalar(x,y,M,N);
    if(d == 3) return cube_fmodf_scalar(x,y,M,N,O);
    if(d == 4) return field_fmodf_scalar(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_fmodf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fmodf_scalar(x,y,output,M);
    if(d == 2) matrix_r_fmodf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_fmodf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_fmodf_scalar(x,y,output,M,N,O,P);
        
}


float* array_remainderf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_remainderf_scalar(x,y,M);
    if(d == 2) return matrix_remainderf_scalar(x,y,M,N);
    if(d == 3) return cube_remainderf_scalar(x,y,M,N,O);
    if(d == 4) return field_remainderf_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_remainderf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_remainderf_scalar(x,y,output,M);
    if(d == 2) matrix_r_remainderf_scalar(x,y,output,M,N);
    if(d == 3) cube_r_remainderf_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_remainderf_scalar(x,y,output,M,N,O,P);
        
}

float* array_atan2f_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) return vector_atan2f_scalar(x,y,M);
    if(d == 2) return matrix_atan2f_scalar(x,y,M,N);
    if(d == 3) return cube_atan2f_scalar(x,y,M,N,O);
    if(d == 4) return field_atan2f_scalar(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_atan2f_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_atan2f_scalar(x,y,output,M);
    if(d == 2) matrix_r_atan2f_scalar(x,y,output,M,N);
    if(d == 3) cube_r_atan2f_scalar(x,y,output,M,N,O);
    if(d == 4) field_r_atan2f_scalar(x,y,output,M,N,O,P);
        
}

float* array_cbrtf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_cbrtf(x,M);
    if(d == 2) return matrix_cbrtf(x,M,N);
    if(d == 3) return cube_cbrtf(x,M,N,O);
    if(d == 4) return field_cbrtf(x,M,N,O,P);
    
    return NULL;
}

void array_r_cbrtf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_cbrtf(x,output,M);
    if(d == 2) matrix_r_cbrtf(x,output,M,N);
    if(d == 3) cube_r_cbrtf(x,output,M,N,O);
    if(d == 4) field_r_cbrtf(x,output,M,N,O,P);
        
}



float* array_copysignf(float * x, float *y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_copysignf(x,y,M);
    if(d == 2) return matrix_copysignf(x,y,M,N);
    if(d == 3) return cube_copysignf(x,y,M,N,O);
    if(d == 4) return field_copysignf(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_copysignf(float * x, float *y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_copysignf(x,y,output,M);
    if(d == 2) matrix_r_copysignf(x,y,output,M,N);
    if(d == 3) cube_r_copysignf(x,y,output,M,N,O);
    if(d == 4) field_r_copysignf(x,y,output,M,N,O,P);
    
}
float* array_cospif(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_cospif(x,M);
    if(d == 2) return matrix_cospif(x,M,N);
    if(d == 3) return cube_cospif(x,M,N,O);
    if(d == 4) return field_cospif(x,M,N,O,P);
    
    return NULL;
}

void array_r_cospif(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_cospif(x,output,M);
    if(d == 2) matrix_r_cospif(x,output,M,N);
    if(d == 3) cube_r_cospif(x,output,M,N,O);
    if(d == 4) field_r_cospif(x,output,M,N,O,P);
    
}

float* array_cyl_bessel_i0f(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_cyl_bessel_i0f(x,M);
    if(d == 2) return matrix_cyl_bessel_i0f(x,M,N);
    if(d == 3) return cube_cyl_bessel_i0f(x,M,N,O);
    if(d == 4) return field_cyl_bessel_i0f(x,M,N,O,P);
    
    return NULL;
}

void array_r_cyl_bessel_i0f(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_cyl_bessel_i0f(x,output,M);
    if(d == 2) matrix_r_cyl_bessel_i0f(x,output,M,N);
    if(d == 3) cube_r_cyl_bessel_i0f(x,output,M,N,O);
    if(d == 4) field_r_cyl_bessel_i0f(x,output,M,N,O,P);
    
}

float* array_cyl_bessel_i1f(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_cyl_bessel_i1f(x,M);
    if(d == 2) return matrix_cyl_bessel_i1f(x,M,N);
    if(d == 3) return cube_cyl_bessel_i1f(x,M,N,O);
    if(d == 4) return field_cyl_bessel_i1f(x,M,N,O,P);
    
    return NULL;
}

void array_r_cyl_bessel_i1f(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_cyl_bessel_i1f(x,output,M);
    if(d == 2) matrix_r_cyl_bessel_i1f(x,output,M,N);
    if(d == 3) cube_r_cyl_bessel_i1f(x,output,M,N,O);
    if(d == 4) field_r_cyl_bessel_i1f(x,output,M,N,O,P);
    
}

float* array_erfcf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_erfcf(x,M);
    if(d == 2) return matrix_erfcf(x,M,N);
    if(d == 3) return cube_erfcf(x,M,N,O);
    if(d == 4) return field_erfcf(x,M,N,O,P);
    
    return NULL;
}

void array_r_erfcf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_erfcf(x,output,M);
    if(d == 2) matrix_r_erfcf(x,output,M,N);
    if(d == 3) cube_r_erfcf(x,output,M,N,O);
    if(d == 4) field_r_erfcf(x,output,M,N,O,P);
    
}

float* array_erfcinvf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_erfcinvf(x,M);
    if(d == 2) return matrix_erfcinvf(x,M,N);
    if(d == 3) return cube_erfcinvf(x,M,N,O);
    if(d == 4) return field_erfcinvf(x,M,N,O,P);
    
    return NULL;
}

void array_r_erfcinvf(float * x, float * output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_erfcinvf(x,output,M);
    if(d == 2) matrix_r_erfcinvf(x,output,M,N);
    if(d == 3) cube_r_erfcinvf(x,output,M,N,O);
    if(d == 4) field_r_erfcinvf(x,output,M,N,O,P);
    
}

float* array_erfcxf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_erfcxf(x,M);
    if(d == 2) return matrix_erfcxf(x,M,N);
    if(d == 3) return cube_erfcxf(x,M,N,O);
    if(d == 4) return field_erfcxf(x,M,N,O,P);
    
    return NULL;
}

void array_r_erfcxf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_erfcxf(x,output,M);
    if(d == 2) matrix_r_erfcxf(x,output,M,N);
    if(d == 3) cube_r_erfcxf(x,output,M,N,O);
    if(d == 4) field_r_erfcxf(x,output,M,N,O,P);
    
}


float* array_erff(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_erff(x,M);
    if(d == 2) return matrix_erff(x,M,N);
    if(d == 3) return cube_erff(x,M,N,O);
    if(d == 4) return field_erff(x,M,N,O,P);
    
    return NULL;
}

void array_r_erff(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_erff(x,output,M);
    if(d == 2) matrix_r_erff(x,output,M,N);
    if(d == 3) cube_r_erff(x,output,M,N,O);
    if(d == 4) field_r_erff(x,output,M,N,O,P);
    
}

float* array_erfinvf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_erfinvf(x,M);
    if(d == 2) return matrix_erfinvf(x,M,N);
    if(d == 3) return cube_erfinvf(x,M,N,O);
    if(d == 4) return field_erfinvf(x,M,N,O,P);
    
    return NULL;
}

void array_r_erfinvf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_erfinvf(x,output,M);
    if(d == 2) matrix_r_erfinvf(x,output,M,N);
    if(d == 3) cube_r_erfinvf(x,output,M,N,O);
    if(d == 4) field_r_erfinvf(x,output,M,N,O,P);
    
}

float* array_fdimf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fdimf(x,y,M);
    if(d == 2) return matrix_fdimf(x,y,M,N);
    if(d == 3) return cube_fdimf(x,y,M,N,O);
    if(d == 4) return field_fdimf(x,y,M,N,O,P);
        
    
    return NULL;
}

void array_r_fdimf(float * x, float * y, float * output, int M, int N, int O, int P)
{    
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fdimf(x,y,output,M);
    if(d == 2) matrix_r_fdimf(x,y,output,M,N);
    if(d == 3) cube_r_fdimf(x,y,output,M,N,O);
    if(d == 4) field_r_fdimf(x,y,output,M,N,O,P);
    
}

float* array_fdividef(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fdividef(x,y,M);
    if(d == 2) return matrix_fdividef(x,y,M,N);
    if(d == 3) return cube_fdividef(x,y,M,N,O);
    if(d == 4) return field_fdividef(x,y,M,N,O,P);
        
    
    return NULL;
}

void array_r_fdividef(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fdividef(x,y,output,M);
    if(d == 2) matrix_r_fdividef(x,y,output,M,N);
    if(d == 3) cube_r_fdividef(x,y,output,M,N,O);
    if(d == 4) field_r_fdividef(x,y,output,M,N,O,P);
        
}

float* array_fmaf(float * x, float * y, float * z, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fmaf(x,y,z,M);
    if(d == 2) return matrix_fmaf(x,y,z,M,N);
    if(d == 3) return cube_fmaf(x,y,z,M,N,O);
    if(d == 4) return field_fmaf(x,y,z,M,N,O,P);
    
    return NULL;
}

void array_r_fmaf(float * x, float * y, float * z, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fmaf(x,y,z,output,M);
    if(d == 2) matrix_r_fmaf(x,y,z,output,M,N);
    if(d == 3) cube_r_fmaf(x,y,z,output,M,N,O);
    if(d == 4) field_r_fmaf(x,y,z,output,M,N,O,P);
        
}

float* array_hypotf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_hypotf(x,y,M);
    if(d == 2) return matrix_hypotf(x,y,M,N);
    if(d == 3) return cube_hypotf(x,y,M,N,O);
    if(d == 4) return field_hypotf(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_hypotf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_hypotf(x,y,output,M);
    if(d == 2) matrix_r_hypotf(x,y,output,M,N);
    if(d == 3) cube_r_hypotf(x,y,output,M,N,O);
    if(d == 4) field_r_hypotf(x,y,output,M,N,O,P);
        
}

float* array_ilogbf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_ilogbf(x,M);
    if(d == 2) return matrix_ilogbf(x,M,N);
    if(d == 3) return cube_ilogbf(x,M,N,O);
    if(d == 4) return field_ilogbf(x,M,N,O,P);
    
    return NULL;
}

void array_r_ilogbf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_ilogbf(x,output,M);
    if(d == 2) matrix_r_ilogbf(x,output,M,N);
    if(d == 3) cube_r_ilogbf(x,output,M,N,O);
    if(d == 4) field_r_ilogbf(x,output,M,N,O,P);
        
}

float* array_j0f(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_j0f(x,M);
    if(d == 2) return matrix_j0f(x,M,N);
    if(d == 3) return cube_j0f(x,M,N,O);
    if(d == 4) return field_j0f(x,M,N,O,P);
    
    return NULL;
}

void array_r_j0f(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_j0f(x,output,M);
    if(d == 2) matrix_r_j0f(x,output,M,N);
    if(d == 3) cube_r_j0f(x,output,M,N,O);
    if(d == 4) field_r_j0f(x,output,M,N,O,P);
        
}

float* array_j1f(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_j1f(x,M);
    if(d == 2) return matrix_j1f(x,M,N);
    if(d == 3) return cube_j1f(x,M,N,O);
    if(d == 4) return field_j1f(x,M,N,O,P);
    
    return NULL;
}

void array_r_j1f(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_j1f(x,output,M);
    if(d == 2) matrix_r_j1f(x,output,M,N);
    if(d == 3) cube_r_j1f(x,output,M,N,O);
    if(d == 4) field_r_j1f(x,output,M,N,O,P);
        
}
float* array_jnf(float * x, int m, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_jnf(x,m,M);
    if(d == 2) return matrix_jnf(x,m,M,N);
    if(d == 3) return cube_jnf(x,m,M,N,O);
    if(d == 4) return field_jnf(x,m,M,N,O,P);
    
    return NULL;
}

void array_r_jnf(float * x, float *output, int m, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_jnf(x,output,m,M);
    if(d == 2) matrix_r_jnf(x,output,m,M,N);
    if(d == 3) cube_r_jnf(x,output,m,M,N,O);
    if(d == 4) field_r_jnf(x,output,m,M,N,O,P);
        
}

float* array_ldexpf(float * x,int exp, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_ldexpf(x,exp,M);
    if(d == 2) return matrix_ldexpf(x,exp,M,N);
    if(d == 3) return cube_ldexpf(x,exp,M,N,O);
    if(d == 4) return field_ldexpf(x,exp,M,N,O,P);
    
    return NULL;
}

void array_r_ldexpf(float * x, float * output, int exp, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_ldexpf(x,output,exp,M);
    if(d == 2) matrix_r_ldexpf(x,output,exp,M,N);
    if(d == 3) cube_r_ldexpf(x,output,exp,M,N,O);
    if(d == 4) field_r_ldexpf(x,output,exp,M,N,O,P);
        
}

float* array_lgammaf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_lgammaf(x,M);
    if(d == 2) return matrix_lgammaf(x,M,N);
    if(d == 3) return cube_lgammaf(x,M,N,O);
    if(d == 4) return field_lgammaf(x,M,N,O,P);
    
    return NULL;
}

void array_r_lgammaf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_lgammaf(x,output,M);
    if(d == 2) matrix_r_lgammaf(x,output,M,N);
    if(d == 3) cube_r_lgammaf(x,output,M,N,O);
    if(d == 4) field_r_lgammaf(x,output,M,N,O,P);
        
}

float* array_nearbyintf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_nearbyintf(x,M);
    if(d == 2) return matrix_nearbyintf(x,M,N);
    if(d == 3) return cube_nearbyintf(x,M,N,O);
    if(d == 4) return field_nearbyintf(x,M,N,O,P);
    
    return NULL;

}
void array_r_nearbyintf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_nearbyintf(x,output,M);
    if(d == 2) matrix_r_nearbyintf(x,output,M,N);
    if(d == 3) cube_r_nearbyintf(x,output,M,N,O);
    if(d == 4) field_r_nearbyintf(x,output,M,N,O,P);
        
}

float* array_norm3df(float * x, float * y, float * z, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_norm3df(x,y,z,M);
    if(d == 2) return matrix_norm3df(x,y,z,M,N);
    if(d == 3) return cube_norm3df(x,y,z,M,N,O);
    if(d == 4) return field_norm3df(x,y,z,M,N,O,P);
    
    return NULL;
}

void array_r_norm3df(float * x, float * y, float * z, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_norm3df(x,y,z,output,M);
    if(d == 2) matrix_r_norm3df(x,y,z,output,M,N);
    if(d == 3) cube_r_norm3df(x,y,z,output,M,N,O);
    if(d == 4) field_r_norm3df(x,y,z,output,M,N,O,P);
        
}

float* array_norm4df(float * x, float * y, float * z, float * q, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_norm4df(x,y,z,q,M);
    if(d == 2) return matrix_norm4df(x,y,z,q,M,N);
    if(d == 3) return cube_norm4df(x,y,z,q,M,N,O);
    if(d == 4) return field_norm4df(x,y,z,q,M,N,O,P);
    
    return NULL;
}

void array_r_norm4df(float * x, float * y, float * z, float * q, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_norm4df(x,y,z,q,output,M);
    if(d == 2) matrix_r_norm4df(x,y,z,q,output,M,N);
    if(d == 3) cube_r_norm4df(x,y,z,q,output,M,N,O);
    if(d == 4) field_r_norm4df(x,y,z,q,output,M,N,O,P);
        
}

float* array_normcdff(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_normcdff(x,M);
    if(d == 2) return matrix_normcdff(x,M,N);
    if(d == 3) return cube_normcdff(x,M,N,O);
    if(d == 4) return field_normcdff(x,M,N,O,P);
    
    return NULL;
}

void array_r_normcdff(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_normcdff(x,output,M);
    if(d == 2) matrix_r_normcdff(x,output,M,N);
    if(d == 3) cube_r_normcdff(x,output,M,N,O);
    if(d == 4) field_r_normcdff(x,output,M,N,O,P);
        
}

float* array_normcdfinvf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_normcdfinvf(x,M);
    if(d == 2) return matrix_normcdfinvf(x,M,N);
    if(d == 3) return cube_normcdfinvf(x,M,N,O);
    if(d == 4) return field_normcdfinvf(x,M,N,O,P);
    
    return NULL;
}

void array_r_normcdfinvf(float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_normcdfinvf(x,output,M);
    if(d == 2) matrix_r_normcdfinvf(x,output,M,N);
    if(d == 3) cube_r_normcdfinvf(x,output,M,N,O);
    if(d == 4) field_r_normcdfinvf(x,output,M,N,O,P);
        
}

float* array_normf(int dim, float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_normf(dim,x,M);
    if(d == 2) return matrix_normf(dim,x,M,N);
    if(d == 3) return cube_normf(dim,x,M,N,O);
    if(d == 4) return field_normf(dim,x,M,N,O,P);
    
    return NULL;
}

void array_r_normf(int dim, float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_normf(dim,x,output,M);
    if(d == 2) matrix_r_normf(dim,x,output,M,N);
    if(d == 3) cube_r_normf(dim,x,output,M,N,O);
    if(d == 4) field_r_normf(dim,x,output,M,N,O,P);
        
}


float* array_rnorm3df(float * x, float * y, float * z, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_rnorm3df(x,y,z,M);
    if(d == 2) return matrix_rnorm3df(x,y,z,M,N);
    if(d == 3) return cube_rnorm3df(x,y,z,M,N,O);
    if(d == 4) return field_rnorm3df(x,y,z,M,N,O,P);
    
    return NULL;
}

void array_r_rnorm3df(float * x, float * y, float * z, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_rnorm3df(x,y,z,output,M);
    if(d == 2) matrix_r_rnorm3df(x,y,z,output,M,N);
    if(d == 3) cube_r_rnorm3df(x,y,z,output,M,N,O);
    if(d == 4) field_r_rnorm3df(x,y,z,output,M,N,O,P);
        
}

float* array_rnorm4df(float * x, float * y, float * z, float * q, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_rnorm4df(x,y,z,q,M);
    if(d == 2) return matrix_rnorm4df(x,y,z,q,M,N);
    if(d == 3) return cube_rnorm4df(x,y,z,q,M,N,O);
    if(d == 4) return field_rnorm4df(x,y,z,q,M,N,O,P);
    
    return NULL;
}

void array_r_rnorm4df(float * x, float * y, float * z, float * q, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_rnorm4df(x,y,z,q,output,M);
    if(d == 2) matrix_r_rnorm4df(x,y,z,q,output,M,N);
    if(d == 3) cube_r_rnorm4df(x,y,z,q,output,M,N,O);
    if(d == 4) field_r_rnorm4df(x,y,z,q,output,M,N,O,P);
        
}

float* array_rnormf(int dim, float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_rnormf(dim,x,M);
    if(d == 2) return matrix_rnormf(dim,x,M,N);
    if(d == 3) return cube_rnormf(dim,x,M,N,O);
    if(d == 4) return field_rnormf(dim,x,M,N,O,P);
    
    return NULL;
}

void array_r_rnormf(int dim, float * x, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_rnormf(dim,x,output,M);
    if(d == 2) matrix_r_rnormf(dim,x,output,M,N);
    if(d == 3) cube_r_rnormf(dim,x,output,M,N,O);
    if(d == 4) field_r_rnormf(dim,x,output,M,N,O,P);
        
}

float* array_rcbrtf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_rcbrtf(x,M);
    if(d == 2) return matrix_rcbrtf(x,M,N);
    if(d == 3) return cube_rcbrtf(x,M,N,O);
    if(d == 4) return field_rcbrtf(x,M,N,O,P);
    
    return NULL;
}

void array_r_rcbrtf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_rcbrtf(x,output,M);
    if(d == 2) matrix_r_rcbrtf(x,output,M,N);
    if(d == 3) cube_r_rcbrtf(x,output,M,N,O);
    if(d == 4) field_r_rcbrtf(x,output,M,N,O,P);
        
}

float* array_remainderf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_remainderf(x,y,M);
    if(d == 2) return matrix_remainderf(x,y,M,N);
    if(d == 3) return cube_remainderf(x,y,M,N,O);
    if(d == 4) return field_remainderf(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_remainderf(float * x, float * y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_remainderf(x,y,output,M);
    if(d == 2) matrix_r_remainderf(x,y,output,M,N);
    if(d == 3) cube_r_remainderf(x,y,output,M,N,O);
    if(d == 4) field_r_remainderf(x,y,output,M,N,O,P);
        
}

float* array_rhypotf(float * x, float * y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_rhypotf(x,y,M);
    if(d == 2) return matrix_rhypotf(x,y,M,N);
    if(d == 3) return cube_rhypotf(x,y,M,N,O);
    if(d == 4) return field_rhypotf(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_rhypotf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_rhypotf(x,y,output,M);
    if(d == 2) matrix_r_rhypotf(x,y,output,M,N);
    if(d == 3) cube_r_rhypotf(x,y,output,M,N,O);
    if(d == 4) field_r_rhypotf(x,y,output,M,N,O,P);
        
}

float* array_scalblnf(float * x, long int _M, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);    
    if(d == 1) return vector_scalblnf(x,_M,M);
    if(d == 2) return matrix_scalblnf(x,_M,M,N);
    if(d == 3) return cube_scalblnf(x,_M,M,N,O);
    if(d == 4) return field_scalblnf(x,_M,M,N,O,P);
    
    return NULL;
}

void array_r_scalblnf(float * x, long int _M, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_scalblnf(x,_M,output,M);
    if(d == 2) matrix_r_scalblnf(x,_M,output,M,N);
    if(d == 3) cube_r_scalblnf(x,_M,output,M,N,O);
    if(d == 4) field_r_scalblnf(x,_M,output,M,N,O,P);
        
}

float* array_sinpif(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);    
    if(d == 1) return vector_sinpif(x,M);
    if(d == 2) return matrix_sinpif(x,M,N);
    if(d == 3) return cube_sinpif(x,M,N,O);
    if(d == 4) return field_sinpif(x,M,N,O,P);
    
    return NULL;
}

void array_r_sinpif(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_sinpif(x,output,M);
    if(d == 2) matrix_r_sinpif(x,output,M,N);
    if(d == 3) cube_r_sinpif(x,output,M,N,O);
    if(d == 4) field_r_sinpif(x,output,M,N,O,P);
        
}

float* array_tgammaf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);    
    if(d == 1) return vector_tgammaf(x,M);
    if(d == 2) return matrix_tgammaf(x,M,N);
    if(d == 3) return cube_tgammaf(x,M,N,O);
    if(d == 4) return field_tgammaf(x,M,N,O,P);
    
    return NULL;
}

void array_r_tgammaf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_tgammaf(x,output,M);
    if(d == 2) matrix_r_tgammaf(x,output,M,N);
    if(d == 3) cube_r_tgammaf(x,output,M,N,O);
    if(d == 4) field_r_tgammaf(x,output,M,N,O,P);
        
}

float* array_truncf(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);    
    if(d == 1) return vector_truncf(x,M);
    if(d == 2) return matrix_truncf(x,M,N);
    if(d == 3) return cube_truncf(x,M,N,O);
    if(d == 4) return field_truncf(x,M,N,O,P);
    
    return NULL;
}

void array_r_truncf(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_truncf(x,output,M);
    if(d == 2) matrix_r_truncf(x,output,M,N);
    if(d == 3) cube_r_truncf(x,output,M,N,O);
    if(d == 4) field_r_truncf(x,output,M,N,O,P);
        
}

float* array_y0f(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);    
    if(d == 1) return vector_y0f(x,M);
    if(d == 2) return matrix_y0f(x,M,N);
    if(d == 3) return cube_y0f(x,M,N,O);
    if(d == 4) return field_y0f(x,M,N,O,P);
    
    return NULL;
}
void array_r_y0f(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_y0f(x,output,M);
    if(d == 2) matrix_r_y0f(x,output,M,N);
    if(d == 3) cube_r_y0f(x,output,M,N,O);
    if(d == 4) field_r_y0f(x,output,M,N,O,P);
        
}

float* array_y1f(float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);    
    if(d == 1) return vector_y1f(x,M);
    if(d == 2) return matrix_y1f(x,M,N);
    if(d == 3) return cube_y1f(x,M,N,O);
    if(d == 4) return field_y1f(x,M,N,O,P);
    
    return NULL;
}
void array_r_y1f(float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_y1f(x,output,M);
    if(d == 2) matrix_r_y1f(x,output,M,N);
    if(d == 3) cube_r_y1f(x,output,M,N,O);
    if(d == 4) field_r_y1f(x,output,M,N,O,P);
        
}

float* array_ynf(int m,float * x, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);    
    if(d == 1) return vector_ynf(m,x,M);
    if(d == 2) return matrix_ynf(m,x,M,N);
    if(d == 3) return cube_ynf(m,x,M,N,O);
    if(d == 4) return field_ynf(m,x,M,N,O,P);
    
    return NULL;
}
void array_r_ynf(int m, float * x, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_ynf(m,x,output,M);
    if(d == 2) matrix_r_ynf(m,x,output,M,N);
    if(d == 3) cube_r_ynf(m,x,output,M,N,O);
    if(d == 4) field_r_ynf(m,x,output,M,N,O,P);
        
}

float* array_fdimf_const(float * x, float  y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fdimf_const(x,y,M);
    if(d == 2) return matrix_fdimf_const(x,y,M,N);
    if(d == 3) return cube_fdimf_const(x,y,M,N,O);
    if(d == 4) return field_fdimf_const(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_fdimf_const(float * x, float  y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fdimf_const(x,y,output,M);
    if(d == 2) matrix_r_fdimf_const(x,y,output,M,N);
    if(d == 3) cube_r_fdimf_const(x,y,output,M,N,O);
    if(d == 4) field_r_fdimf_const(x,y,output,M,N,O,P);
            
}


float* array_fdividef_const(float * x, float y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_fdividef_const(x,y,M);
    if(d == 2) return matrix_fdividef_const(x,y,M,N);
    if(d == 3) return cube_fdividef_const(x,y,M,N,O);
    if(d == 4) return field_fdividef_const(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_fdividef_const(float * x, float y, float * output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_fdividef_const(x,y,output,M);
    if(d == 2) matrix_r_fdividef_const(x,y,output,M,N);
    if(d == 3) cube_r_fdividef_const(x,y,output,M,N,O);
    if(d == 4) field_r_fdividef_const(x,y,output,M,N,O,P);
                
}


float* array_hypotf_const(float * x, float y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_hypotf_const(x,y,M);
    if(d == 2) return matrix_hypotf_const(x,y,M,N);
    if(d == 3) return cube_hypotf_const(x,y,M,N,O);
    if(d == 4) return field_hypotf_const(x,y,M,N,O,P);
    
    return NULL;
}
void array_r_hypotf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_hypotf_const(x,y,output,M);
    if(d == 2) matrix_r_hypotf_const(x,y,output,M,N);
    if(d == 3) cube_r_hypotf_const(x,y,output,M,N,O);
    if(d == 4) field_r_hypotf_const(x,y,output,M,N,O,P);
                
}



float* array_remainderf_const(float * x, float y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_remainderf_const(x,y,M);
    if(d == 2) return matrix_remainderf_const(x,y,M,N);
    if(d == 3) return cube_remainderf_const(x,y,M,N,O);
    if(d == 4) return field_remainderf_const(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_remainderf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_remainderf_const(x,y,output,M);
    if(d == 2) matrix_r_remainderf_const(x,y,output,M,N);
    if(d == 3) cube_r_remainderf_const(x,y,output,M,N,O);
    if(d == 4) field_r_remainderf_const(x,y,output,M,N,O,P);
                
}


float* array_rhypotf_const(float * x, float y, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) return vector_rhypotf_const(x,y,M);
    if(d == 2) return matrix_rhypotf_const(x,y,M,N);
    if(d == 3) return cube_rhypotf_const(x,y,M,N,O);
    if(d == 4) return field_rhypotf_const(x,y,M,N,O,P);
    
    return NULL;
}

void array_r_rhypotf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int d = dims(M,N,O,P);
    if(d == 1) vector_r_rhypotf_const(x,y,output,M);
    if(d == 2) matrix_r_rhypotf_const(x,y,output,M,N);
    if(d == 3) cube_r_rhypotf_const(x,y,output,M,N,O);
    if(d == 4) field_r_rhypotf_const(x,y,output,M,N,O,P);
                
}
