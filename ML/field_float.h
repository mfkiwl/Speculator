#ifndef __FLOATFIELD_H
#define __FLOATFIELD_H

// these can be C callable.
#ifdef __cplusplus
extern "C" {
#endif 

float* field_addf(float * x, float * y, int M, int N, int O, int P);
float* field_subf(float * x, float * y, int M, int N, int O, int P);
float* field_mulf(float * x, float * y, int M, int N, int O, int P);
float* field_divf(float * x, float * y, int M, int N, int O, int P);
float* field_modf(float * x, float * y, int M, int N, int O, int P);

float* field_acosf(float * devPtr, int M, int N, int O, int P);
float* field_acoshf(float * devPtr, int M, int N, int O, int P);
float* field_asinf(float * devPtr, int M, int N, int O, int P);
float* field_asinhf(float * devPtr, int M, int N, int O, int P);
float* field_atan2f(float * a, float * b, int M, int N, int O, int P);
float* field_atanf(float * devPtr, int M, int N, int O, int P);
float* field_atanhf(float * devPtr, int M, int N, int O, int P);
float* field_cbrtf(float * devPtr, int M, int N, int O, int P);
float* field_ceilf(float * devPtr, int M, int N, int O, int P);
float* field_cosf(float * devPtr, int M, int N, int O, int P);
float* field_coshf(float * devPtr, int M, int N, int O, int P);
float* field_exp10f(float * devPtr, int M, int N, int O, int P);
float* field_exp2f(float * devPtr, int M, int N, int O, int P);
float* field_expf(float * devPtr, int M, int N, int O, int P);
float* field_expm1f(float * devPtr, int M, int N, int O, int P);
float* field_fabsf(float * devPtr, int M, int N, int O, int P);
float* field_floorf(float * devPtr, int M, int N, int O, int P);
float* field_fmaxf(float * x, float * y, int M, int N, int O, int P);
float* field_fminf(float * x, float * y, int M, int N, int O, int P);
float* field_fmodf(float * x, float * y, int M, int N, int O, int P);
float* field_hypotf(float * x, float * y, int M, int N, int O, int P);
float* field_log10f(float * x, int M, int N, int O, int P);
float* field_log1pf(float * x, int M, int N, int O, int P);
float* field_log2f(float * x, int M, int N, int O, int P);
float* field_logbf(float * x, int M, int N, int O, int P);
float* field_powf(float * x, float * y, int M, int N, int O, int P);
float* field_rsqrtf(float * x, int M, int N, int O, int P);
float* field_sinf(float * x, int M, int N, int O, int P);
float* field_sinhf(float * x, int M, int N, int O, int P);
float* field_sinpif(float * x, int M, int N, int O, int P);
float* field_sqrtf(float * x, int M, int N, int O, int P);
float* field_tanf(float * x, int M, int N, int O, int P);
float* field_tanhf(float * x, int M, int N, int O, int P);

float* field_sigmoidf(float * devPtr, int M, int N, int O, int P);
float* field_sigmoid_gradf(float * devPtr, int M, int N, int O, int P);
float* field_tanh_gradf(float * devPtr, int M, int N, int O, int P);
float* field_reluf(float * devPtr, int M, int N, int O, int P);
float* field_relu_gradf(float * devPtr, int M, int N, int O, int P);
float* field_softmaxf(float * x, int M, int N, int O, int P);

float* field_addf_const(float * x, float  y, int M, int N, int O, int P);
float* field_subf_const(float * x, float  y, int M, int N, int O, int P);
float* field_mulf_const(float * x, float  y, int M, int N, int O, int P);
float* field_divf_const(float * x, float  y, int M, int N, int O, int P);
float* field_modf_const(float * x, float  y, int M, int N, int O, int P);
float* field_atan2f_const(float * a, float b, int M, int N, int O, int P);
float* field_fmaxf_const(float * x, float  y, int M, int N, int O, int P);
float* field_fminf_const(float * x, float  y, int M, int N, int O, int P);
float* field_fmodf_const(float * x, float  y, int M, int N, int O, int P);
float* field_powf_const(float * x, float y, int M, int N, int O, int P);


float* field_addf_scalar(float * x, float * y, int M, int N, int O, int P);
float* field_subf_scalar(float * x, float * y, int M, int N, int O, int P);
float* field_mulf_scalar(float * x, float * y, int M, int N, int O, int P);
float* field_divf_scalar(float * x, float * y, int M, int N, int O, int P);
float* field_modf_scalar(float * x, float * y, int M, int N, int O, int P);
float* field_atan2f_scalar(float * a, float *b, int M, int N, int O, int P);
float* field_fmaxf_scalar(float * x, float  *y, int M, int N, int O, int P);
float* field_fminf_scalar(float * x, float  *y, int M, int N, int O, int P);
float* field_fmodf_scalar(float * x, float  *y, int M, int N, int O, int P);
float* field_powf_scalar(float * x, float *y, int M, int N, int O, int P);


float* field_copysignf(float * X, float *Y, int M, int N, int O, int P);
float* field_cospif(float * devPtr, int M, int N, int O, int P);
float* field_cyl_bessel_i0f(float * devPtr, int M, int N, int O, int P);
float* field_cyl_bessel_i1f(float * devPtr, int M, int N, int O, int P);
float* field_erfcf(float * devPtr, int M, int N, int O, int P);
float* field_erfcinvf(float * devPtr, int M, int N, int O, int P);
float* field_erfcxf(float * devPtr, int M, int N, int O, int P);
float* field_erff(float * devPtr, int M, int N, int O, int P);
float* field_erfinvf(float * devPtr, int M, int N, int O, int P);
float* field_fdimf(float * a, float * b, int M, int N, int O, int P);
float* field_fdividef(float * a, float * b, int M, int N, int O, int P);
float* field_fmaf(float * x, float * y, float * z, int M, int N, int O, int P);
float* field_ilogbf(float * x, int M, int N, int O, int P);
float* field_j0f(float * x, int M, int N, int O, int P);
float* field_j1f(float * x, int M, int N, int O, int P);
float* field_jnf(float * x, int m, int M, int N, int O, int P);
float* field_ldexpf(float * x, int exp, int M, int N, int O, int P);
float* field_lgammaf(float * x, int M, int N, int O, int P);
long long* field_llrintf(float * x, int M, int N, int O, int P);
long long* field_llroundf(float * x, int M, int N, int O, int P);
long* field_lrintf(float * x, int M, int N, int O, int P);
long* field_lroundf(float * x, int M, int N, int O, int P);
float* field_nearbyintf(float * x, int M, int N, int O, int P);
float* field_norm3df(float * x, float * y, float * z, int M, int N, int O, int P);
float* field_norm4df(float * x, float * y, float * z, float * q, int M, int N, int O, int P);
float* field_normcdff(float * x, int M, int N, int O, int P);
float* field_normcdfinvf(float * x, int M, int N, int O, int P);
float* field_normf(int dim, float * x, int M, int N, int O, int P);
float* field_rcbrtf(float * x, int M, int N, int O, int P);
float* field_remainderf(float * x, float * y, int M, int N, int O, int P);
float* field_rhypotf(float * x, float * y, int M, int N, int O, int P);
float* field_rnorm3df(float * x, float * y, float * z, int M, int N, int O, int P);
float* field_rnorm4df(float * x, float * y, float * z, float * q, int M, int N, int O, int P);
float* field_rnormf(int dim, float * x, int M, int N, int O, int P);
float* field_tgammaf(float * x, int M, int N, int O, int P);
float* field_y0f(float * x, int M, int N, int O, int P);
float* field_y1f(float * x, int M, int N, int O, int P);
float* field_ynf(int n, float * x, int M, int N, int O, int P);
float* field_tgammaf(float * x, int M, int N, int O, int P);
float* field_y0f(float * x, int M, int N, int O, int P);
float* field_y1f(float * x, int M, int N, int O, int P);
float* field_ynf(int n, float * x, int M, int N, int O, int P);
float* field_scalblnf(float * x, long int m, int M, int N, int O, int P);

float* field_fdimf_const(float * a, float  b, int M, int N, int O, int P);
float* field_fdividef_const(float * a, float  b, int M, int N, int O, int P);
float* field_hypotf_const(float * x, float  y, int M, int N, int O, int P);
float* field_remainderf_const(float * x, float y, int M, int N, int O, int P);
float* field_rhypotf_const(float * x, float y, int M, int N, int O, int P);

float* field_fdimf_scalar(float * a, float  *b, int M, int N, int O, int P);
float* field_fdividef_scalar(float * a, float *b, int M, int N, int O, int P);
float* field_hypotf_scalar(float * x, float  *y, int M, int N, int O, int P);
float* field_remainderf_scalar(float * x, float *y, int M, int N, int O, int P);
float* field_rhypotf_scalar(float * x, float *y, int M, int N, int O, int P);


void field_addf_row(float * x, int row, float * y, size_t n);
void field_subf_row(float * x, int row, float * y, size_t n);
void field_mulf_row(float * x, int row, float * y, size_t n);
void field_divf_row(float * x, int row, float * y, size_t n);
void field_modf_row(float * x, int row, float * y, size_t n);

void field_r_addf(float * x, float * y, float * output, int M, int N, int O, int P);
void field_r_subf(float * x, float * y, float * output, int M, int N, int O, int P);
void field_r_mulf(float * x, float * y, float * output, int M, int N, int O, int P);
void field_r_divf(float * x, float * y, float * output, int M, int N, int O, int P);
void field_r_modf(float * x, float * y, float * output, int M, int N, int O, int P);
void field_r_acosf(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_asinf(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_atanf(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_atan2f(float * a, float * b, float * output, int M, int N, int O, int P);
void field_r_acoshf(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_asinhf(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_atanhf(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_cosf(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_sinf(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_tanf(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_coshf(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_sinhf(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_tanhf(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_ceilf(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_exp10f(float * devPtr, float * outputs, int M, int N, int O, int P);
void field_r_exp2f(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_expf(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_expm1f(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_fabsf(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_floorf(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_fmaxf(float * x, float * y, float * output, int M, int N, int O, int P);
void field_r_fminf(float * x, float * y, float * output, int M, int N, int O, int P);
void field_r_fmodf(float * x, float * y, float * output, int M, int N, int O, int P);
void field_r_log10f(float * x, float *output, int M, int N, int O, int P);
void field_r_log1pf(float * x, float *output, int M, int N, int O, int P);
void field_r_log2f(float * x, float * output, int M, int N, int O, int P);
void field_r_logbf(float * x, float *output, int M, int N, int O, int P);
void field_r_powf(float * x, float * y, float *output, int M, int N, int O, int P);
void field_r_rsqrtf(float * x, float * output, int M, int N, int O, int P);
void field_r_sinf(float * x, float *output, int M, int N, int O, int P);
void field_r_sinhf(float * x, float *output, int M, int N, int O, int P);
void field_r_sqrtf(float * x, float *output, int M, int N, int O, int P);
void field_r_tanf(float * x, float *output, int M, int N, int O, int P);
void field_r_tanhf(float * x, float *output, int M, int N, int O, int P);
void field_r_softmaxf(float * x, float *output, int M, int N, int O, int P);
void field_r_sigmoidf(float * x, float *output, int M, int N, int O, int P);
void field_r_sigmoid_gradf(float * x, float *output, int M, int N, int O, int P);
void field_r_tanh_gradf(float * x, float *output, int M, int N, int O, int P);
void field_r_reluf(float * x, float *output, int M, int N, int O, int P);
void field_r_relu_gradf(float * x, float *output, int M, int N, int O, int P);
void field_r_cbrtf(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_cospif(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_cyl_bessel_i0f(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_cyl_bessel_i1f(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_erfcf(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_erfcinvf(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_erfcxf(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_erff(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_erfinvf(float * devPtr, float * output, int M, int N, int O, int P);
void field_r_fdimf(float * a, float * b, float * output, int M, int N, int O, int P);
void field_r_fdividef(float * a, float * b, float * output, int M, int N, int O, int P);
void field_r_fmaf(float * x, float * y, float * z, float *output, int M, int N, int O, int P);
void field_r_hypotf(float * x, float * y, float * output, int M, int N, int O, int P);
void field_r_ilogbf(float * x, float *output, int M, int N, int O, int P);
void field_r_j0f(float * x, float *output, int M, int N, int O, int P);
void field_r_j1f(float * x, float *output, int M, int N, int O, int P);
void field_r_jnf(float * x, float * output, int m, int M, int N, int O, int P);
void field_r_ldexpf(float * x, float * output, int exp, int M, int N, int O, int P);
void field_r_lgammaf(float * x, float *output, int M, int N, int O, int P);
void field_r_nearbyintf(float * x, float *output, int M, int N, int O, int P);
void field_r_norm3df(float * x, float * y, float * z, float * output, int M, int N, int O, int P);
void field_r_norm4df(float * x, float * y, float * z, float * q, float * output, int M, int N, int O, int P);
void field_r_normcdff(float * x, float * output, int M, int N, int O, int P);
void field_r_normcdfinvf(float * x, float *output, int M, int N, int O, int P);
void field_r_normf(int dim, float * x, float * output, int M, int N, int O, int P);
void field_r_rcbrtf(float * x, float *output, int M, int N, int O, int P);
void field_r_remainderf(float * x, float * y, float *output, int M, int N, int O, int P);
void field_r_rhypotf(float * x, float * y, float *output, int M, int N, int O, int P);
void field_r_rnorm3df(float * x, float * y, float * z, float * output, int M, int N, int O, int P);
void field_r_rnorm4df(float * x, float * y, float * z, float * q, float *output, int M, int N, int O, int P);
void field_r_rnormf(int dim, float * x, float *output, int M, int N, int O, int P);
void field_r_scalblnf(float * x, long int n, float * output, int M, int N, int O, int P);
void field_r_tgammaf(float * x, float * output, int M, int N, int O, int P);
void field_r_truncf(float * x, float *output, int M, int N, int O, int P);
void field_r_y0f(float * x, float *output, int M, int N, int O, int P);
void field_r_y1f(float * x, float * output, int M, int N, int O, int P);
void field_r_ynf(int n, float * x, float *output, int M, int N, int O, int P);
void field_r_sinpif(float * x, float *output, int M, int N, int O, int P);


void field_r_addf_const(float * x, float  y, float *output, int M, int N, int O, int P);
void field_r_subf_const(float * x, float  y, float *output,int M, int N, int O, int P);
void field_r_mulf_const(float * x, float  y, float *output,int M, int N, int O, int P);
void field_r_divf_const(float * x, float  y, float *output,int M, int N, int O, int P);
void field_r_modf_const(float * x, float  y, float *output,int M, int N, int O, int P);
void field_r_atan2f_const(float * a, float b, float *output,int M, int N, int O, int P);
void field_r_fmaxf_const(float * x, float  y, float *output,int M, int N, int O, int P);
void field_r_fminf_const(float * x, float  y, float *output,int M, int N, int O, int P);
void field_r_fmodf_const(float * x, float  y, float *output,int M, int N, int O, int P);
void field_r_powf_const(float * x, float y, float *output,int M, int N, int O, int P);


void field_r_addf_scalar(float * x, float * y, float *output,int M, int N, int O, int P);
void field_r_subf_scalar(float * x, float * y, float *output,int M, int N, int O, int P);
void field_r_mulf_scalar(float * x, float * y, float *output,int M, int N, int O, int P);
void field_r_divf_scalar(float * x, float * y, float *output,int M, int N, int O, int P);
void field_r_modf_scalar(float * x, float * y, float *output,int M, int N, int O, int P);
void field_r_atan2f_scalar(float * a, float *b, float *output,int M, int N, int O, int P);
void field_r_fmaxf_scalar(float * x, float  *y, float *output,int M, int N, int O, int P);
void field_r_fminf_scalar(float * x, float  *y, float *output,int M, int N, int O, int P);
void field_r_fmodf_scalar(float * x, float  *y, float *output,int M, int N, int O, int P);
void field_r_powf_scalar(float * x, float *y, float *output,int M, int N, int O, int P);

void field_r_fdimf_const(float * a, float  b, float *output,int M, int N, int O, int P);
void field_r_fdividef_const(float * a, float  b, float *output,int M, int N, int O, int P);
void field_r_hypotf_const(float * x, float  y, float *output,int M, int N, int O, int P);
void field_r_remainderf_const(float * x, float y, float *output,int M, int N, int O, int P);
void field_r_rhypotf_const(float * x, float y, float *output,int M, int N, int O, int P);

void field_r_fdimf_scalar(float * a, float  *b, float *output,int M, int N, int O, int P);
void field_r_fdividef_scalar(float * a, float *b, float *output,int M, int N, int O, int P);
void field_r_hypotf_scalar(float * x, float  *y, float *output,int M, int N, int O, int P);
void field_r_remainderf_scalar(float * x, float *y, float *output,int M, int N, int O, int P);
void field_r_rhypotf_scalar(float * x, float *y, float *output,int M, int N, int O, int P);


float* field_truncf(float * x, int M, int N, int O, int P);
void field_r_truncf(float * x, float *output, int M, int N, int O, int P);

void field_r_copysignf(float * X, float *Y, float *output, int M, int N, int O, int P);

#ifdef __cplusplus
}
#endif 


#endif