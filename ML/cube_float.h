#ifndef __FLOATCUBE_H
#define __FLOATCUBE_H


// these can be C callable.
#ifdef __cplusplus
extern "C" {
#endif 

float* cube_addf(float * x, float * y, int M, int N, int O);
float* cube_subf(float * x, float * y, int M, int N, int O);
float* cube_mulf(float * x, float * y, int M, int N, int O);
float* cube_divf(float * x, float * y, int M, int N, int O);
float* cube_modf(float * x, float * y, int M, int N, int O);

float* cube_acosf(float * devPtr, int M, int N, int O);
float* cube_acosf(float * devPtr, int M, int N, int O);
float* cube_asinf(float * devPtr, int M, int N, int O);
float* cube_atan2f(float * a, float * b, int M, int N, int O);
float* cube_atanf(float * devPtr, int M, int N, int O);
float* cube_acoshf(float * devPtr, int M, int N, int O);
float* cube_asinhf(float * devPtr, int M, int N, int O);
float* cube_atanhf(float * devPtr, int M, int N, int O);
float* cube_cbrtf(float * devPtr, int M, int N, int O);
float* cube_ceilf(float * devPtr, int M, int N, int O);
float* cube_cosf(float * devPtr, int M, int N, int O);
float* cube_coshf(float * devPtr, int M, int N, int O);
float* cube_exp10f(float * devPtr, int M, int N, int O);
float* cube_exp2f(float * devPtr, int M, int N, int O);
float* cube_expf(float * devPtr, int M, int N, int O);
float* cube_expm1f(float * devPtr, int M, int N, int O);
float* cube_fabsf(float * devPtr, int M, int N, int O);
float* cube_floorf(float * devPtr, int M, int N, int O);
float* cube_fmaxf(float * x, float * y, int M, int N, int O);
float* cube_fminf(float * x, float * y, int M, int N, int O);
float* cube_fmodf(float * x, float * y, int M, int N, int O);
float* cube_hypotf(float * x, float * y, int M, int N, int O);
float* cube_log10f(float * x, int M, int N, int O);
float* cube_log1pf(float * x, int M, int N, int O);
float* cube_log2f(float * x, int M, int N, int O);
float* cube_logbf(float * x, int M, int N, int O);
float* cube_powf(float * x, float * y, int M, int N, int O);
float* cube_rsqrtf(float * x, int M, int N, int O);
float* cube_sinf(float * x, int M, int N, int O);
float* cube_sinhf(float * x, int M, int N, int O);
float* cube_sinpif(float * x, int M, int N, int O);
float* cube_sqrtf(float * x, int M, int N, int O);
float* cube_tanf(float * x, int M, int N, int O);
float* cube_tanhf(float * x, int M, int N, int O);

float* cube_sigmoidf(float * devPtr, int M, int N, int O);
float* cube_sigmoid_gradf(float * devPtr, int M, int N, int O);
float* cube_tanh_gradf(float * devPtr, int M, int N, int O);
float* cube_reluf(float * devPtr, int M, int N, int O);
float* cube_relu_gradf(float * devPtr, int M, int N, int O);
float* cube_softmaxf(float * x, int M, int N, int O);

float* cube_addf_const(float * x, float  y, int M, int N, int O);
float* cube_subf_const(float * x, float  y, int M, int N, int O);
float* cube_mulf_const(float * x, float  y, int M, int N, int O);
float* cube_divf_const(float * x, float  y, int M, int N, int O);
float* cube_modf_const(float * x, float  y, int M, int N, int O);
float* cube_atan2f_const(float * a, float b, int M, int N, int O);
float* cube_fmaxf_const(float * x, float  y, int M, int N, int O);
float* cube_fminf_const(float * x, float  y, int M, int N, int O);
float* cube_fmodf_const(float * x, float  y, int M, int N, int O);
float* cube_powf_const(float * x, float y, int M, int N, int O);


float* cube_addf_scalar(float * x, float * y, int M, int N, int O);
float* cube_subf_scalar(float * x, float * y, int M, int N, int O);
float* cube_mulf_scalar(float * x, float * y, int M, int N, int O);
float* cube_divf_scalar(float * x, float * y, int M, int N, int O);
float* cube_modf_scalar(float * x, float * y, int M, int N, int O);
float* cube_atan2f_scalar(float * a, float *b, int M, int N, int O);
float* cube_fmaxf_scalar(float * x, float  *y, int M, int N, int O);
float* cube_fminf_scalar(float * x, float  *y, int M, int N, int O);
float* cube_fmodf_scalar(float * x, float  *y, int M, int N, int O);
float* cube_powf_scalar(float * x, float *y, int M, int N, int O);


float* cube_copysignf(float * X, float *Y, int M, int N, int O);
float* cube_cospif(float * devPtr, int M, int N, int O);
float* cube_cyl_bessel_i0f(float * devPtr, int M, int N, int O);
float* cube_cyl_bessel_i1f(float * devPtr, int M, int N, int O);
float* cube_erfcf(float * devPtr, int M, int N, int O);
float* cube_erfcinvf(float * devPtr, int M, int N, int O);
float* cube_erfcxf(float * devPtr, int M, int N, int O);
float* cube_erff(float * devPtr, int M, int N, int O);
float* cube_erfinvf(float * devPtr, int M, int N, int O);
float* cube_fdimf(float * a, float * b, int M, int N, int O);
float* cube_fdividef(float * a, float * b, int M, int N, int O);
float* cube_fmaf(float * x, float * y, float * z, int M, int N, int O);
float* cube_ilogbf(float * x, int M, int N, int O);
float* cube_j0f(float * x, int M, int N, int O);
float* cube_j1f(float * x, int M, int N, int O);
float* cube_jnf(float * x, int m, int M, int N, int O);
float* cube_ldexpf(float * x, int exp, int M, int N, int O);
float* cube_lgammaf(float * x, int M, int N, int O);
long long* cube_llrintf(float * x, int M, int N, int O);
long long* cube_llroundf(float * x, int M, int N, int O);
long* cube_lrintf(float * x, int M, int N, int O);
long* cube_lroundf(float * x, int M, int N, int O);
float* cube_nearbyintf(float * x, int M, int N, int O);
float* cube_norm3df(float * x, float * y, float * z, int M, int N, int O);
float* cube_norm4df(float * x, float * y, float * z, float * q, int M, int N, int O);
float* cube_normcdff(float * x, int M, int N, int O);
float* cube_normcdfinvf(float * x, int M, int N, int O);
float* cube_normf(int dim, float * x, int M, int N, int O);
float* cube_rcbrtf(float * x, int M, int N, int O);
float* cube_remainderf(float * x, float * y, int M, int N, int O);
float* cube_rhypotf(float * x, float * y, int M, int N, int O);
float* cube_rnorm3df(float * x, float * y, float * z, int M, int N, int O);
float* cube_rnorm4df(float * x, float * y, float * z, float * q, int M, int N, int O);
float* cube_rnormf(int dim, float * x, int M, int N, int O);
float* cube_tgammaf(float * x, int M, int N, int O);
float* cube_y0f(float * x, int M, int N, int O);
float* cube_y1f(float * x, int M, int N, int O);
float* cube_ynf(int n, float * x, int M, int N, int O);
float* cube_tgammaf(float * x, int M, int N, int O);
float* cube_y0f(float * x, int M, int N, int O);
float* cube_y1f(float * x, int M, int N, int O);
float* cube_ynf(int n, float * x, int M, int N, int O);

float* cube_fdimf_const(float * a, float  b, int M, int N, int O);
float* cube_fdividef_const(float * a, float  b, int M, int N, int O);
float* cube_hypotf_const(float * x, float  y, int M, int N, int O);
float* cube_remainderf_const(float * x, float y, int M, int N, int O);
float* cube_rhypotf_const(float * x, float y, int M, int N, int O);

float* cube_fdimf_scalar(float * a, float  *b, int M, int N, int O);
float* cube_fdividef_scalar(float * a, float *b, int M, int N, int O);
float* cube_hypotf_scalar(float * x, float  *y, int M, int N, int O);
float* cube_remainderf_scalar(float * x, float *y, int M, int N, int O);
float* cube_rhypotf_scalar(float * x, float *y, int M, int N, int O);


void cube_addf_row(float * x, int row, float * y, size_t n);
void cube_subf_row(float * x, int row, float * y, size_t n);
void cube_mulf_row(float * x, int row, float * y, size_t n);
void cube_divf_row(float * x, int row, float * y, size_t n);
void cube_modf_row(float * x, int row, float * y, size_t n);

void cube_r_addf(float * x, float * y, float * output, int M, int N, int O);
void cube_r_subf(float * x, float * y, float * output, int M, int N, int O);
void cube_r_mulf(float * x, float * y, float * output, int M, int N, int O);
void cube_r_divf(float * x, float * y, float * output, int M, int N, int O);
void cube_r_modf(float * x, float * y, float * output, int M, int N, int O);
void cube_r_acosf(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_asinf(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_atanf(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_atan2f(float * a, float * b, float * output, int M, int N, int O);
void cube_r_acoshf(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_asinhf(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_atanhf(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_cosf(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_sinf(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_tanf(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_coshf(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_sinhf(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_tanhf(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_ceilf(float * devPtr, float * output, int M, int N, int O);
void cube_r_exp10f(float * devPtr, float * outputs, int M, int N, int O);
void cube_r_exp2f(float * devPtr, float * output, int M, int N, int O);
void cube_r_expf(float * devPtr, float * output, int M, int N, int O);
void cube_r_expm1f(float * devPtr, float * output, int M, int N, int O);
void cube_r_fabsf(float * devPtr, float * output, int M, int N, int O);
void cube_r_floorf(float * devPtr, float * output, int M, int N, int O);
void cube_r_fmaxf(float * x, float * y, float * output, int M, int N, int O);
void cube_r_fminf(float * x, float * y, float * output, int M, int N, int O);
void cube_r_fmodf(float * x, float * y, float * output, int M, int N, int O);
void cube_r_log10f(float * x, float *output, int M, int N, int O);
void cube_r_log1pf(float * x, float *output, int M, int N, int O);
void cube_r_log2f(float * x, float * output, int M, int N, int O);
void cube_r_logbf(float * x, float *output, int M, int N, int O);
void cube_r_powf(float * x, float * y, float *output, int M, int N, int O);
void cube_r_rsqrtf(float * x, float * output, int M, int N, int O);
void cube_r_sinf(float * x, float *output, int M, int N, int O);
void cube_r_sinhf(float * x, float *output, int M, int N, int O);
void cube_r_sqrtf(float * x, float *output, int M, int N, int O);
void cube_r_tanf(float * x, float *output, int M, int N, int O);
void cube_r_tanhf(float * x, float *output, int M, int N, int O);
void cube_r_softmaxf(float * x, float *output, int M, int N, int O);
void cube_r_sigmoidf(float * x, float *output, int M, int N, int O);
void cube_r_sigmoid_gradf(float * x, float *output, int M, int N, int O);
void cube_r_tanh_gradf(float * x, float *output, int M, int N, int O);
void cube_r_reluf(float * x, float *output, int M, int N, int O);
void cube_r_relu_gradf(float * x, float *output, int M, int N, int O);
void cube_r_cbrtf(float * devPtr, float * output, int M, int N, int O);
void cube_r_cospif(float * devPtr, float * output, int M, int N, int O);
void cube_r_cyl_bessel_i0f(float * devPtr, float * output, int M, int N, int O);
void cube_r_cyl_bessel_i1f(float * devPtr, float * output, int M, int N, int O);
void cube_r_erfcf(float * devPtr, float * output, int M, int N, int O);
void cube_r_erfcinvf(float * devPtr, float * output, int M, int N, int O);
void cube_r_erfcxf(float * devPtr, float * output, int M, int N, int O);
void cube_r_erff(float * devPtr, float * output, int M, int N, int O);
void cube_r_erfinvf(float * devPtr, float * output, int M, int N, int O);
void cube_r_fdimf(float * a, float * b, float * output, int M, int N, int O);
void cube_r_fdividef(float * a, float * b, float * output, int M, int N, int O);
void cube_r_fmaf(float * x, float * y, float * z, float *output, int M, int N, int O);
void cube_r_hypotf(float * x, float * y, float * output, int M, int N, int O);
void cube_r_ilogbf(float * x, float *output, int M, int N, int O);
void cube_r_j0f(float * x, float *output, int M, int N, int O);
void cube_r_j1f(float * x, float *output, int M, int N, int O);
void cube_r_jnf(float * x, float * output, int m, int M, int N, int O);
void cube_r_ldexpf(float * x, float * output, int exp, int M, int N, int O);
void cube_r_lgammaf(float * x, float *output, int M, int N, int O);
void cube_r_nearbyintf(float * x, float *output, int M, int N, int O);
void cube_r_norm3df(float * x, float * y, float * z, float * output, int M, int N, int O);
void cube_r_norm4df(float * x, float * y, float * z, float * q, float * output, int M, int N, int O);
void cube_r_normcdff(float * x, float * output, int M, int N, int O);
void cube_r_normcdfinvf(float * x, float *output, int M, int N, int O);
void cube_r_normf(int dim, float * x, float * output, int M, int N, int O);
void cube_r_rcbrtf(float * x, float *output, int M, int N, int O);
void cube_r_remainderf(float * x, float * y, float *output, int M, int N, int O);
void cube_r_rhypotf(float * x, float * y, float *output, int M, int N, int O);
void cube_r_rnorm3df(float * x, float * y, float * z, float * output, int M, int N, int O);
void cube_r_rnorm4df(float * x, float * y, float * z, float * q, float *output, int M, int N, int O);
void cube_r_rnormf(int dim, float * x, float *output, int M, int N, int O);
void cube_r_scalblnf(float * x, long int n, float * output, int M, int N, int O);
void cube_r_tgammaf(float * x, float * output, int M, int N, int O);
void cube_r_truncf(float * x, float *output, int M, int N, int O);
void cube_r_y0f(float * x, float *output, int M, int N, int O);
void cube_r_y1f(float * x, float * output, int M, int N, int O);
void cube_r_ynf(int n, float * x, float *output, int M, int N, int O);
void cube_r_sinpif(float * x, float *output, int M, int N, int O);


void cube_r_addf_const(float * x, float  y, float *output, int M, int N, int O);
void cube_r_subf_const(float * x, float  y, float *output,int M, int N, int O);
void cube_r_mulf_const(float * x, float  y, float *output,int M, int N, int O);
void cube_r_divf_const(float * x, float  y, float *output,int M, int N, int O);
void cube_r_modf_const(float * x, float  y, float *output,int M, int N, int O);
void cube_r_atan2f_const(float * a, float b, float *output,int M, int N, int O);
void cube_r_fmaxf_const(float * x, float  y, float *output,int M, int N, int O);
void cube_r_fminf_const(float * x, float  y, float *output,int M, int N, int O);
void cube_r_fmodf_const(float * x, float  y, float *output,int M, int N, int O);
void cube_r_powf_const(float * x, float y, float *output,int M, int N, int O);


void cube_r_addf_scalar(float * x, float * y, float *output,int M, int N, int O);
void cube_r_subf_scalar(float * x, float * y, float *output,int M, int N, int O);
void cube_r_mulf_scalar(float * x, float * y, float *output,int M, int N, int O);
void cube_r_divf_scalar(float * x, float * y, float *output,int M, int N, int O);
void cube_r_modf_scalar(float * x, float * y, float *output,int M, int N, int O);
void cube_r_atan2f_scalar(float * a, float *b, float *output,int M, int N, int O);
void cube_r_fmaxf_scalar(float * x, float  *y, float *output,int M, int N, int O);
void cube_r_fminf_scalar(float * x, float  *y, float *output,int M, int N, int O);
void cube_r_fmodf_scalar(float * x, float  *y, float *output,int M, int N, int O);
void cube_r_powf_scalar(float * x, float *y, float *output,int M, int N, int O);

void cube_r_fdimf_const(float * a, float  b, float *output,int M, int N, int O);
void cube_r_fdividef_const(float * a, float  b, float *output,int M, int N, int O);
void cube_r_remainderf_const(float * x, float y, float *output,int M, int N, int O);
void cube_r_hypotf_const(float * x, float y, float *output,int M, int N, int O);
void cube_r_rhypotf_const(float * x, float y, float *output,int M, int N, int O);

void cube_r_fdimf_scalar(float * a, float  *b, float *output,int M, int N, int O);
void cube_r_fdividef_scalar(float * a, float *b, float *output,int M, int N, int O);
void cube_r_remainderf_scalar(float * x, float *y, float *output,int M, int N, int O);
void cube_r_hypotf_scalar(float * x, float  *y, float *output,int M, int N, int O);
void cube_r_rhypotf_scalar(float * x, float *y, float *output,int M, int N, int O);

float* cube_scalblnf(float * x, long int m, int M, int N, int O);

float* cube_truncf(float * x, int M, int N, int O);
void cube_r_truncf(float * x, float *output, int M, int N, int O);

void cube_r_copysignf(float * X, float *Y, float *output, int M, int N, int O);

#ifdef __cplusplus
}
#endif 

#endif