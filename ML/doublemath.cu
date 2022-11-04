////////////////////////////////////////////////////////////////////////
// vector
////////////////////////////////////////////////////////////////////////


#include <cassert>
#include "cuda_runtime.h"
#include "math_constants.h"
#include <map>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "vector_double.h"


cudaStream_t get_cuda_stream();
cudaStream_t random_stream();

// memory cache
std::multimap<int,double*> cuda_memory;

void add_memory(int length, double * f) {
    cuda_memory.insert(std::pair<int,double*>(length,f));
}
void return_memory(int length, double *fp) {        
    cuda_memory.insert(std::pair<int,double*>(length,fp));    
}
double* find_memory(int length) {            
    typename std::multimap<int,double*>::iterator i = cuda_memory.find(length);
    if(i == cuda_memory.end()) return NULL;                
    cuda_zero(i->second,length);
    cuda_memory.erase(i);
    return i->second;
}
void clear_cache() {
    typename std::multimap<int,double*>::iterator i = cuda_memory.begin();
    while(i != cuda_memory.end()) {
        cudaFree(i->second);       
        i++; 
    }
    cuda_memory.clear();    
}


/* what we want is this 
__global__ void vector_program(double * x, double * out, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    out[idx] = eval(x)
}

////////////////////////////////////
Vector X;
Matrix M;
Vector output;

f(x) = M*X;
////////////////////////////////////

=>
    int row = threadIdx.x + blockIdx.x * blockDim.x;     
    int col = threadIdx.y + blockIdx.y * blockDim.y;     
    if(row < m && col < n)
        out[col] = M[row*n + col] * X[col]

*/

/*
std::map<std::string,vector_kernel1> vector_map1;
std::map<std::string,vector_kernel2> vector_map2;
std::map<std::string,vector_kernel3> vector_map3;
std::map<std::string,vector_kernel4> vector_map4;

void register_vector_kernel1(const char * name, vector_kernel1 kernel) {
    // should assert if already exist?
    vector_map1[name] = kernel;
}
void register_vector_kernel2(const char * name, vector_kernel2 kernel) {
    vector_map2[name] = kernel;
}
void register_vector_kernel3(const char * name, vector_kernel3 kernel) {
    vector_map3[name] = kernel;
}
void register_vector_kernel4(const char * name, vector_kernel4 kernel) {
    vector_map4[name] = kernel;
}
double* execute_vector_kernel1(const char * name, double * x, int n) {
    typename std::map<std::string,vector_kernel1>::iterator i = vector_map1.find(name);
    // assert or return NULL?
    if(i == vector_map1.end()) return NULL;
    return (i->second)(x,n);
}
double* execute_vector_kernel2(const char * name, double * x, double * y, int n) {
    typename std::map<std::string,vector_kernel2>::iterator i = vector_map2.find(name);
    // assert or return NULL?
    if(i == vector_map2.end()) return NULL;
    return (i->second)(x,y,n);
}
double* execute_vector_kernel3(const char * name, double * x, double * y, double * z, int n) {
    typename std::map<std::string,vector_kernel3>::iterator i = vector_map3.find(name);
    // assert or return NULL?
    if(i == vector_map3.end()) return NULL;
    return (i->second)(x,y,z,n);
}
double* execute_vector_kernel4(const char * name, double * x, double * y, double * z, double * w, int n) {
    typename std::map<std::string,vector_kernel4>::iterator i = vector_map4.find(name);
    // assert or return NULL?
    if(i == vector_map4.end()) return NULL;
    return (i->second)(x,y,z,w,n);
}
*/


__global__ void vector_dummy(double * x, double * out, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] =x[idx];
}


void calcSize(int N,int * gridSize, int * blockSize) {
    
    //int minGridSize = 0;    
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, blockSize, vector_dummy, 0, N); 
    //*gridSize = (N + *blockSize - 1) / *blockSize; 
    
    
    *blockSize=1024;    
    *gridSize=(N+*blockSize)/ *blockSize;    
}

__global__ void cuda_memcpy_device(double * dst, double * src, size_t N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) dst[idx] = src[idx];
}

void cuda_memcpy(double * dst, double * src, int n)
{    
    //int gridSize,blockSize;
    //calcSize(n,&gridSize,&blockSize);        
    //cuda_memcpy_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(dst,src,n);        
    //cudaDeviceSynchronize();
    cudaMemcpyAsync(dst,src,n*sizeof(double),cudaMemcpyDeviceToDevice, get_cuda_stream());
}

__global__ void cuda_zero_device(double * dst, size_t N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) dst[idx] = 0;
}

void cuda_zero(double * dst, int n)
{    
    /*
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    cuda_zero_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(dst,n);        
    */    
    cudaMemsetAsync(dst,0,n*sizeof(double), get_cuda_stream());    
}



/// making this parallel is alot harder than you would think it should be.
/// I need atomics so I can add all at once.
__global__ void vector_sum_device(double * x,double *out, int N)
{
    
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    //if(idx < N) *out += x[idx];
    for(int i = 0; i < N; i++) 
    {
        *out += x[i];
        __syncthreads();
    }
}

double vector_sumf(double * x, int n)
{        
    /*
    double * output = find_memory(1);
    if(output == NULL) cudaMalloc((void**)&output,sizeof(double));    
    double zero = 0.0f;
    cudaMemcpyAsync(output,&zero,sizeof(double),cudaMemcpyHostToDevice);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_sum_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);        
    double o=0;
    cudaMemcpyAsync(&o,output,sizeof(double),cudaMemcpyDeviceToHost);
    return_memory(1,output);
    */
    double *tmp = (double*)calloc(n,sizeof(double));
    cudaMemcpyAsync(tmp,x,n*sizeof(double),cudaMemcpyDeviceToHost, get_cuda_stream());
    double o = 0.0;
    for(size_t i = 0; i < n; i++) o += tmp[i];
    free(tmp);    
    return o;
}

__global__ void vector_prod_device(double * x,double *out, int N)
{
    for(int i = 0; i < N; i++) {
        *out *= x[i];
        __syncthreads();
    }
}

double vector_prodf(double * x, int n)
{
    /*
    double * output = find_memory(1);
    if(output == NULL) cudaMalloc((void**)&output,sizeof(double));    
    double zero = 1.0f;
    cudaMemcpyAsync(output,&zero,sizeof(double),cudaMemcpyHostToDevice);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_prod_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);        
    double o=0;
    cudaMemcpyAsync(&o,output,sizeof(double),cudaMemcpyDeviceToHost);
    return_memory(1,output);
    */
    double *tmp = (double*)calloc(n,sizeof(double));
    cudaMemcpyAsync(tmp,x,n*sizeof(double),cudaMemcpyDeviceToHost, get_cuda_stream());
    double o = 1.0;
    for(size_t i = 0; i < n; i++) o *= tmp[i];
    free(tmp);    
    return o;
}



__global__ void vector_addf_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] + y[idx];
}

double* vector_addf(double * x, double * y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_addf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);    
    return output;
}


void vector_r_addf(double * x, double * y, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_addf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);    
}



__global__ void vector_subf_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] - y[idx];
}

double* vector_subf(double * x, double * y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_subf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);    
    return output;
}


void vector_r_subf(double * x, double * y, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_subf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);    
}



__global__ void vector_mulf_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] * y[idx];
}

double* vector_mulf(double * x, double * y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_mulf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);    
    return output;
}

void vector_r_mulf(double * x, double * y, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_mulf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);    
}

__global__ void vector_divf_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] / y[idx];
}

double* vector_divf(double * x, double * y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_divf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);    
    return output;
}
void vector_r_divf(double * x, double * y, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_divf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);    
}

__global__ void vector_modf_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = fmodf(x[idx],y[idx]);
}

double* vector_modf(double * x, double * y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_modf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);    
    return output;
}
void vector_r_modf(double * x, double * y, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_modf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);    
}



__global__ void vector_acosf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n) out[idx] = acosf(in[idx]);
}

double* vector_acosf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_acosf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);    
    return output;
}
void vector_r_acosf(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_acosf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);        
}

__global__ void vector_acoshf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n) out[idx] = acoshf(in[idx]);
}

double* vector_acoshf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_acoshf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}

void vector_r_acoshf(double * devPtr, double * output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_acoshf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n); 
}

__global__ void vector_asinhf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = asinhf(in[idx]);
}

double* vector_asinhf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_asinhf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_asinhf(double * devPtr, double * output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_asinhf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n); 
}


__global__ void vector_asinf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = asinf(in[idx]);
}

double* vector_asinf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_asinf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}

void vector_r_asinf(double * devPtr, double * output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_asinf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}


__global__ void vector_atan2f_device(double * a, double * b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = atan2f(a[idx],b[idx]);
}

double* vector_atan2f(double * a, double * b, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_atan2f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(a,b,output,n);
    return output;
}

void vector_r_atan2f(double * a, double * b, double * output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_atan2f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(a,b,output,n); 
}



__global__ void vector_atanf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = atanf(in[idx]);
}

double* vector_atanf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_atanf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}

void vector_r_atanf(double * devPtr, double * output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_atanf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);    
}



__global__ void vector_atanhf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = atanhf(in[idx]);
}

double* vector_atanhf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_atanhf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}

void vector_r_atanhf(double * devPtr, double * output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_atanhf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n); 
}



__global__ void vector_ceilf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = ceilf(in[idx]);
}

double* vector_ceilf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));   
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_ceilf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}

void vector_r_ceilf(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_ceilf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}



__global__ void vector_cosf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = cosf(in[idx]);
}

double* vector_cosf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_cosf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_cosf(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_cosf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);    
}


__global__ void vector_coshf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = coshf(in[idx]);
}

double* vector_coshf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_coshf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_coshf(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_coshf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}



__global__ void vector_exp10f_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = exp10f(in[idx]);
}

double* vector_exp10f(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_exp10f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_exp10f(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_exp10f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}


__global__ void vector_exp2f_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = exp2f(in[idx]);
}

double* vector_exp2f(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_exp2f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_exp2f(double * devPtr, double * output, int n)
{   
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_exp2f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n); 
}



__global__ void vector_expf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = expf(in[idx]);
}

double* vector_expf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_expf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_expf(double * devPtr, double * output, int n)
{   
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_expf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n); 
}

__global__ void vector_expm1f_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = expm1f(in[idx]);
}

double* vector_expm1f(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_expm1f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_expm1f(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_expm1f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}


__global__ void vector_fabsf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = fabsf(in[idx]);
}

double* vector_fabsf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fabsf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_fabsf(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fabsf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}

__global__ void vector_floorf_device(double * a,double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = floorf(a[idx]);
}

double* vector_floorf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_floorf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_floorf(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_floorf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);    
}



__global__ void vector_fmaxf_device(double * a,double * b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = fmaxf(a[idx],b[idx]);
}

double* vector_fmaxf(double * x, double * y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmaxf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
    return output;
}

void vector_r_fmaxf(double * x, double * y, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmaxf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
}


__global__ void vector_fminf_device(double * a,double * b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = fminf(a[idx],b[idx]);
}

double* vector_fminf(double * x, double * y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fminf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
    return output;
}

void vector_r_fminf(double * x, double * y, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fminf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
}

__global__ void vector_fmodf_device(double * a,double * b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = fmodf(a[idx],b[idx]);
}

double* vector_fmodf(double * x, double * y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmodf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
    return output;
}
void vector_r_fmodf(double * x, double * y, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmodf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
}

__global__ void vector_log10f_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = log10f(a[idx]);
}

double* vector_log10f(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_log10f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_log10f(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_log10f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}


__global__ void vector_log1pf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = log1pf(a[idx]);
}


double* vector_log1pf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));        
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_log1pf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}

void vector_r_log1pf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_log1pf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}


__global__ void vector_log2f_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = log2f(a[idx]);
}

double* vector_log2f(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_log2f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_log2f(double * x, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_log2f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}


__global__ void vector_logbf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = logbf(a[idx]);
}

double* vector_logbf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_logbf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_logbf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_logbf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}



__global__ void vector_powf_device(double * a, double * b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = powf(a[idx],b[idx]);
}

double* vector_powf(double * x, double * y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_powf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
    return output;
}
void vector_r_powf(double * x, double * y, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_powf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
}




__global__ void vector_rsqrtf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = rsqrtf(a[idx]);
}

double* vector_rsqrtf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rsqrtf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}

void vector_r_rsqrtf(double * x, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rsqrtf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}



__global__ void vector_sinf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = sinf(a[idx]);
}

double* vector_sinf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sinf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_sinf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sinf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}


__global__ void vector_sinhf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = sinhf(a[idx]);
}

double* vector_sinhf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sinhf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_sinhf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sinhf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);    
}

__global__ void vector_sqrtf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = sqrtf(a[idx]);
}

double* vector_sqrtf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sqrtf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}

void vector_r_sqrtf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sqrtf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}


__global__ void vector_tanf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = tanf(a[idx]);
}

double* vector_tanf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_tanf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_tanf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_tanf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}


__global__ void vector_tanhf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = tanhf(a[idx]);
}

double* vector_tanhf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_tanhf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}

void vector_r_tanhf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_tanhf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}




__global__ void vector_softmax_device(double * x,double *out, double sum, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = expf(x[idx]) / sum;
}

double* vector_softmaxf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    double sum = vector_sumf(x,n);
    assert(sum != 0.0);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);
    vector_softmax_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,sum,n);        
    return output;
}
void vector_r_softmaxf(double * x, double *output, int n)
{
    double sum = vector_sumf(x,n);
    assert(sum != 0.0);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);
    vector_softmax_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,sum,n);            
}


__global__ void vector_sigmoid_device(double * x, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = 1.0 / (1.0 + expf(-x[idx]));
}

double* vector_sigmoidf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);            
    vector_sigmoid_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);        
    return output;
}
void vector_r_sigmoidf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);            
    vector_sigmoid_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);        
}


__global__ void vector_sigmoid_grad_device(double * x, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] * (1.0 - x[idx]);
}

double* vector_sigmoid_gradf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sigmoid_grad_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);        
    return output;
}


void vector_r_sigmoid_gradf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sigmoid_grad_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);        
}




__global__ void vector_tanh_grad_device(double * x, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = 1.0 - (x[idx]*x[idx]);
}

double* vector_tanh_gradf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_tanh_grad_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);        
    return output;
}
void vector_r_tanh_gradf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_tanh_grad_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);        
}


__global__ void vector_relu_device(double * x, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) 
    {
        if(x[idx] < 0) out[idx] = 0.0f;
        else out[idx] = x[idx]; 
    }
}

double* vector_reluf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_relu_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);        
    return output;
}

void vector_r_reluf(double * x, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_relu_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);        
}


__global__ void vector_relu_grad_device(double * x, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) 
    {
        if(x[idx] > 0) out[idx] = 1.0;
        else out[idx] = 0.0f;
    }
}

double* vector_relu_gradf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_relu_grad_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);        
    return output;
}
void vector_r_relu_gradf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_relu_grad_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);        
}



__global__ void vector_add_const_device(double * x, double y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] + y;
}

double* vector_addf_const(double * x, double y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_add_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);    
    return output;
}

void vector_r_addf_const(double * x, double y, double * output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_add_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);     
}


__global__ void vector_sub_const_device(double * x, double y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] - y;
}

double* vector_subf_const(double * x, double y, int n)
{  
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sub_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}


void vector_r_subf_const(double * x, double y, double *output, int n)
{  
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sub_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);          
}


__global__ void vector_mul_const_device(double * x, double y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] * y;
}

double* vector_mulf_const(double * x, double y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_mul_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);     
    return output;
}
void vector_r_mulf_const(double * x, double y, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_mul_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}

__global__ void vector_div_const_device(double * x, double y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] / y;
}

double* vector_divf_const(double * x, double y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_div_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_divf_const(double * x, double y, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_div_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}


__global__ void vector_mod_const_device(double * x, double y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = fmodf(x[idx],y);
}

double* vector_modf_const(double * x, double y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_mod_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}

void vector_r_modf_const(double * x, double y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_mod_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}



__global__ void vector_atan2f_const_device(double * a, double b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = atan2f(a[idx],b);
}

double* vector_atan2f_const(double * a, double  b, int n)
{   
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_atan2f_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(a,b,output,n);    
    return output;
}

void vector_r_atan2f_const(double * a, double  b, double * output, int n)
{   
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_atan2f_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(a,b,output,n);    
}





__global__ void vector_fmaxf_const_device(double * a,double b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = fmaxf(a[idx],b);
}

double* vector_fmaxf_const(double * x, double y, int n)
{
    double * p = find_memory(n);
    if(p == NULL)
        cudaMalloc((void**)&p,sizeof(double)*n);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmaxf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);    
    return p;
}

void vector_r_fmaxf_const(double * x, double y, double *p, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmaxf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);        
}


__global__ void vector_fminf_const_device(double * a,double b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = fminf(a[idx],b);
}

double* vector_fminf_const(double * x, double y, int n)
{
    double * p = find_memory(n);
    if(p == NULL)
        cudaMalloc((void**)&p,sizeof(double)*n);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fminf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);    
    return p;
}
void vector_r_fminf_const(double * x, double y, double *p, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fminf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);        
}

__global__ void vector_fmodf_const_device(double * a,double b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = fmodf(a[idx],b);
}

double* vector_fmodf_const(double * x, double y, int n)
{
    double * p = find_memory(n);
    if(p == NULL)
        cudaMalloc((void**)&p,sizeof(double)*n);

    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmodf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);    
    return p;
}

void vector_r_fmodf_const(double * x, double y, double * p, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmodf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);       
}



__global__ void vector_powf_const_device(double * a, double b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = powf(a[idx],b);
}

double* vector_powf_const(double * x, double y, int n)
{
    double * p = find_memory(n);
    if(p == NULL)
        cudaMalloc((void**)&p,sizeof(double)*n);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_powf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);    
    return p;
}
void vector_r_powf_const(double * x, double y, double *p, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_powf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);    
}






/////////////////////////////////
// const/scalar
/////////////////////////////////
__global__ void vector_add_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] + y[0];
}



double* vector_addf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_add_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_addf_scalar(double * x, double * y, double *output, int n)
{      
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_add_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);            
}




__global__ void vector_sub_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] - y[0];
}

double* vector_subf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sub_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_subf_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sub_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}


__global__ void vector_mul_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] * y[0];
}



double* vector_mulf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_mul_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_mulf_scalar(double * x, double * y, double * output, int n)
{        
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_mul_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);            
}


__global__ void vector_div_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N && y[0] != 0.0f) out[idx] = x[idx] / y[0];
}

double* vector_divf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_div_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_divf_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_div_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}



__global__ void vector_mod_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = fmodf(x[idx],y[0]);
}

double* vector_modf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_mod_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}

void vector_r_modf_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_mod_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}

double* vector_fmodf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_mod_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}

void vector_r_fmodf_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_mod_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}



__global__ void vector_fmaxf_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = fmaxf(x[idx],y[0]);
}

double* vector_fmaxf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmaxf_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_fmaxf_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmaxf_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}


__global__ void vector_fminf_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = fminf(x[idx],y[0]);
}

double* vector_fminf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fminf_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_fminf_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fminf_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}

__global__ void vector_powf_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = powf(x[idx],y[0]);
}

double* vector_powf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_powf_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_powf_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_powf_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}

__global__ void vector_atan2f_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = atan2f(x[idx],y[0]);
}

double* vector_atan2f_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_atan2f_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_atan2f_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_atan2f_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}

__global__ void vector_fdimf_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = fdimf(x[idx],y[0]);
}

double* vector_fdimf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fdimf_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_fdimf_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fdimf_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}

__global__ void vector_fdividef_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N && y[0] != 0.0) out[idx] = fdividef(x[idx],y[0]);
}

double* vector_fdividef_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmaxf_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_fdividef_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fdividef_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}

__global__ void vector_remainderf_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = remainderf(x[idx],y[0]);
}

double* vector_remainderf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_remainderf_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_remainderf_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_remainderf_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}

__global__ void vector_hypot_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = hypot(x[idx],y[0]);
}

double* vector_hypotf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_hypot_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_hypotf_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_hypot_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}


__global__ void vector_rhypot_scalar_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = rhypot(x[idx],y[0]);
}

double* vector_rhypotf_scalar(double * x, double * y, int n)
{    
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rhypot_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
    return output;
}
void vector_r_rhypotf_scalar(double * x, double * y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rhypot_scalar_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);        
}





__global__ void vector_setrowf_device(double * dst, double * src, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) dst[idx] = src[idx];
}

void vector_setrowf(double * dst, int dst_row, double * src, int row_src, size_t n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_setrowf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(dst+dst_row*n,src+row_src*n,n);        
}

void vector_copyrowf(double * dst, int dst_row, double * src, int row_src, int n) {
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_setrowf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(dst+dst_row*n,src+row_src*n,n);        
}

__global__ void vector_add_rowf_device(double * x,double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] + y[idx];
}

void vector_addf_row(double * x, int row, double * y, int row_src, size_t n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_add_rowf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x+row,y+row_src*n,x+row,n);        
}

__global__ void vector_sub_rowf_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] - y[idx];
}

void vector_subf_row(double * x, int row, double * y, int row_src, size_t n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_sub_rowf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x+row,y+row_src,x+row,n);        
}

__global__ void vector_mul_rowf_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] * y[idx];    
}

void vector_mulf_row(double * x, int row, double * y, int row_src, size_t n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_mul_rowf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x+row,y+row_src,x+row,n);            

}


__global__ void vector_div_rowf_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = x[idx] / y[idx];
}

void vector_divf_row(double * x,int row, double * y, int row_src, size_t n)
{   
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_div_rowf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x+row,y+row_src,x+row,n);            
}

__global__ void vector_mod_rowf_device(double * x, double * y, double * out, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < N) out[idx] = fmod(x[idx],y[idx]);
}

void vector_modf_row(double * x, int row, double * y, int row_src, size_t n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_mod_rowf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x+row,y+row_src,x+row,n);            
}




// vector.cu
__global__ void vector_cbrtf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = atanf(in[idx]);
}

double* vector_cbrtf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_cbrtf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_cbrtf(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);    
    vector_cbrtf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}


__global__ void vector_copysignf_device(double * x, double * y, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = copysignf(x[idx],y[idx]);
}

double* vector_copysignf(double * X, double *Y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_copysignf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(X,Y,output,n);
    return output;
}
void vector_r_copysignf(double * X, double *Y, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_copysignf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(X,Y,output,n); 
}

__global__ void vector_cospif_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = cospif(in[idx]);
}

double* vector_cospif(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_cospif_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_cospif(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_cospif_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}


__global__ void vector_cyl_bessel_i0f_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = cyl_bessel_i0f(in[idx]);
}

double* vector_cyl_bessel_i0f(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_cyl_bessel_i0f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_cyl_bessel_i0f(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_cyl_bessel_i0f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}

__global__ void vector_cyl_bessel_i1f_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = cyl_bessel_i1f(in[idx]);
}

double* vector_cyl_bessel_i1f(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_cyl_bessel_i1f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_cyl_bessel_i1f(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_cyl_bessel_i1f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}

__global__ void vector_erfcf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = erfcf(in[idx]);
}

double* vector_erfcf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_erfcf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_erfcf(double * devPtr, double * output, int n)
{ 
    int gridSize,blockSize;    
    calcSize(n,&gridSize,&blockSize);        
    vector_erfcf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}



__global__ void vector_erfcinvf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = erfcinvf(in[idx]);
}

double* vector_erfcinvf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_erfcinvf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_erfcinvf(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_erfcinvf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}


__global__ void vector_erfcxf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = erfcxf(in[idx]);
}

double* vector_erfcxf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_erfcxf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_erfcxf(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_erfcxf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}

__global__ void vector_erff_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = erff(in[idx]);
}

double* vector_erff(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_erff_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_erff(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_erff_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}


__global__ void vector_erfinvf_device(double * in, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = erfinvf(in[idx]);
}

double* vector_erfinvf(double * devPtr, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_erfinvf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
    return output;
}
void vector_r_erfinvf(double * devPtr, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_erfinvf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(devPtr,output,n);
}


__global__ void vector_fdimf_device(double * a, double * b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = fdimf(a[idx],b[idx]);
}

double* vector_fdimf(double * a, double * b, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fdimf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(a,b,output,n);
    return output;
}
void vector_r_fdimf(double * a, double * b, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fdimf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(a,b,output,n);
}

__global__ void vector_fdividef_device(double * a, double * b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = fdividef(a[idx],b[idx]);
}

double* vector_fdividef(double * a, double * b, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fdividef_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(a,b,output,n);
    return output;
}
void vector_r_fdividef(double * a, double * b, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fdividef_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(a,b,output,n);
}


__global__ void vector_fmaf_device(double * a, double * b, double * c, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = fmaf(a[idx],b[idx],c[idx]);
}

double* vector_fmaf(double * x, double * y, double * z, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmaf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,z,output,n);
    return output;
}
void vector_r_fmaf(double * x, double * y, double * z, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fmaf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,z,output,n);
}


__global__ void vector_hypotf_device(double * a,double * b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = hypotf(a[idx],b[idx]);
}

double* vector_hypotf(double * x, double * y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_hypotf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
    return output;
}
void vector_r_hypotf(double * x, double * y, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_hypotf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
}


__global__ void vector_ilogbf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = ilogbf(a[idx]);
}

double* vector_ilogbf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_ilogbf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_ilogbf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_ilogbf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}



__global__ void vector_j0f_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = j0f(a[idx]);
}

double* vector_j0f(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_j0f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_j0f(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_j0f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}



__global__ void vector_j1f_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = j1f(a[idx]);
}

double* vector_j1f(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_j1f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_j1f(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_j1f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}

__global__ void vector_jnf_device(double * a, int N, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = jnf(a[idx],N);
}

double* vector_jnf(double * x, int M, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_jnf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,M,output,n);
    return output;
}
void vector_r_jnf(double * x, double * output, int M, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_jnf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,M,output,n);
}

__global__ void vector_ldexpf_device(double * a, int exp, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = ldexpf(a[idx],exp);
}

double* vector_ldexpf(double * x, int exp, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_ldexpf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,exp,output,n);
    return output;
}
void vector_r_ldexpf(double * x, double * output, int exp, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_ldexpf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,exp,output,n);
}



__global__ void vector_lgammaf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = lgammaf(a[idx]);
}

double* vector_lgammaf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_lgammaf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_lgammaf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_lgammaf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}



__global__ void vector_llrintf_device(double * a, long long * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = llrintf(a[idx]);
}

long long* vector_llrintf(double * x, int n)
{
    long long * p;
    cudaMalloc((void**)&p,sizeof(long long)*n);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_llrintf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,p,n);
    return p;
}

__global__ void vector_llroundf_device(double * a, long long * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = llroundf(a[idx]);
}

long long* vector_llroundf(double * x, int n)
{
    long long * p;
    cudaMalloc((void**)&p,sizeof(long long)*n);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_llroundf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,p,n);
    return p;
}

__global__ void vector_lrintf_device(double * a, long * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = lrintf(a[idx]);
}

long * vector_lrintf(double * x, int n)
{
    long * p;
    cudaMalloc((void**)&p,sizeof(long )*n);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_lrintf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,p,n);
    return p;
}

__global__ void vector_lroundf_device(double * a, long * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = lroundf(a[idx]);
}

long * vector_lroundf(double * x, int n)
{
    long * p;
    cudaMalloc((void**)&p,sizeof(long )*n);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_lroundf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,p,n);
    return p;
}

__global__ void vector_nearbyintf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = lroundf(a[idx]);
}

double* vector_nearbyintf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_nearbyintf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_nearbyintf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_nearbyintf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}



__global__ void vector_norm3df_device(double * a, double * b, double * c, double* out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = norm3df(a[idx],b[idx],c[idx]);
}

double* vector_norm3df(double * x, double * y, double * z, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));   
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_norm3df_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,z,output,n);
    return output;
}
void vector_r_norm3df(double * x, double * y, double * z, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_norm3df_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,z,output,n);
}


__global__ void vector_norm4df_device(double * a, double * b, double * c, double * d, double* out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = norm4df(a[idx],b[idx],c[idx],d[idx]);
}

double* vector_norm4df(double * x, double * y, double * z, double * q, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_norm4df_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,z,q,output,n);
    return output;
}
void vector_r_norm4df(double * x, double * y, double * z, double * q, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_norm4df_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,z,q,output,n);
}


__global__ void vector_normcdff_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = normcdff(a[idx]);
}

double* vector_normcdff(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_normcdff_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_normcdff(double * x, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_normcdff_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);    
}


__global__ void vector_normcdfinvf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = normcdfinvf(a[idx]);
}

double* vector_normcdfinvf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_normcdfinvf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_normcdfinvf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_normcdfinvf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}



__global__ void vector_normf_device(int dim, const double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = normf(dim,a);
}

double* vector_normf(int dim, double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));   
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_normf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(dim,x,output,n);
    return output;
}
void vector_r_normf(int dim, double * x, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_normf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(dim,x,output,n);
}

__global__ void vector_rcbrtf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = rcbrtf(a[idx]);
}

double* vector_rcbrtf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rcbrtf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_rcbrtf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rcbrtf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}



__global__ void vector_remainderf_device(double * a, double * b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = remainderf(a[idx],b[idx]);
}


double* vector_remainderf(double * x, double * y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_remainderf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
    return output;
}
void vector_r_remainderf(double * x, double * y, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_remainderf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
}


__global__ void vector_rhypotf_device(double * a, double * b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = rhypotf(a[idx],b[idx]);
}

double* vector_rhypotf(double * x, double * y, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rhypotf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n);
    return output;
}
void vector_r_rhypotf(double * x, double * y, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rhypotf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,output,n); 
}





__global__ void vector_rnorm3df_device(double * a, double * b, double * c, double* out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = rnorm3df(a[idx],b[idx],c[idx]);
}

double* vector_rnorm3df(double * x, double * y, double * z, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rnorm3df_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,z,output,n);
    return output;
}
void vector_r_rnorm3df(double * x, double * y, double * z, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rnorm3df_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,z,output,n);
}

__global__ void vector_rnorm4df_device(double * a, double * b, double * c, double * d, double* out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = rnorm4df(a[idx],b[idx],c[idx],d[idx]);
}


double* vector_rnorm4df(double * x, double * y, double * z, double * q, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rnorm4df_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,z,q,output,n);
    return output;
}
void vector_r_rnorm4df(double * x, double * y, double * z, double * q, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rnorm4df_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,z,q,output,n);
}


__global__ void vector_rnormf_device(int dim, const double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = rnormf(dim,a);
}

double* vector_rnormf(int dim, double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rnormf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(dim,x,output,n);
    return output;
}
void vector_r_rnormf(int dim, double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rnormf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(dim,x,output,n);
}


__global__ void vector_scalblnf_device(double * a, long int N,double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = scalblnf(a[idx],N);
}


double* vector_scalblnf(double * x, long int M, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_scalblnf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,M,output,n);
    return output;
}
void vector_r_scalblnf(double * x, long int M, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_scalblnf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,M,output,n);
}


__global__ void vector_sinpif_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = sinpif(a[idx]);
}

double* vector_sinpif(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sinpif_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_sinpif(double * x, double *output, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_sinpif_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n); 
}



__global__ void vector_tgammaf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = tgammaf(a[idx]);
}

double* vector_tgammaf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_tgammaf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_tgammaf(double * x, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_tgammaf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}




__global__ void vector_truncf_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = truncf(a[idx]);
}

double* vector_truncf(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_truncf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}

void vector_r_truncf(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_truncf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}


__global__ void vector_y0f_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = y0f(a[idx]);
}

double* vector_y0f(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_y0f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_y0f(double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_y0f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}




__global__ void vector_y1f_device(double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = y1f(a[idx]);
}

double* vector_y1f(double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_y1f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
    return output;
}
void vector_r_y1f(double * x, double * output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_y1f_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,output,n);
}

__global__ void vector_ynf_device(int N, double * a, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = ynf(n,a[idx]);
}

double* vector_ynf(int M, double * x, int n)
{
    double * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(double));
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_ynf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(M,x,output,n);
    return output;
}
void vector_r_ynf(int M, double * x, double *output, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_ynf_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(M,x,output,n);
}


__global__ void vector_fdimf_const_device(double * a, double b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = fdimf(a[idx],b);
}

double* vector_fdimf_const(double * a, double  b, int n)
{
    double * p = find_memory(n);    
    if(p == NULL)
        cudaMalloc((void**)&p,sizeof(double)*n);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fdimf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(a,b,p,n);    
    return p;
}
void vector_r_fdimf_const(double * a, double  b, double * output, int n)
{
    double * p = find_memory(n);        
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fdimf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(a,b,output,n);        
}


__global__ void vector_fdividef_const_device(double * a, double b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = fdividef(a[idx],b);
}

double* vector_fdividef_const(double * a, double b, int n)
{
    double * p = find_memory(n);
    if(p == NULL)
        cudaMalloc((void**)&p,sizeof(double)*n);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fdividef_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(a,b,p,n);    
    return p;
}
void vector_r_fdividef_const(double * a, double b, double *p, int n)
{    
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_fdividef_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(a,b,p,n);     
}


__global__ void vector_hypotf_const_device(double * a,double b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = hypotf(a[idx],b);
}

double* vector_hypotf_const(double * x, double y, int n)
{
    double * p = find_memory(n);
    if(p == NULL)
        cudaMalloc((void**)&p,sizeof(double)*n);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_hypotf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);    
    return p;
}
void vector_r_hypotf_const(double * x, double y, double *p, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_hypotf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);    
}



__global__ void vector_remainderf_const_device(double * a, double b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = remainderf(a[idx],b);
}


double* vector_remainderf_const(double * x, double y, int n)
{
    double * p = find_memory(n);
    if(p == NULL)
        cudaMalloc((void**)&p,sizeof(double)*n);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_remainderf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);    
    return p;
}
void vector_r_remainderf_const(double * x, double y, double *p, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_remainderf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);        
}

__global__ void vector_rhypotf_const_device(double * a, double b, double * out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;     
    if(idx < n)
        out[idx] = rhypotf(a[idx],b);
}

double* vector_rhypotf_const(double * x, double y, int n)
{
    double * p = find_memory(n);
    if(p == NULL)
        cudaMalloc((void**)&p,sizeof(double)*n);
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rhypotf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);    
    return p;
}
void vector_r_rhypotf_const(double * x, double y, double *p, int n)
{
    int gridSize,blockSize;
    calcSize(n,&gridSize,&blockSize);        
    vector_rhypotf_const_device<<<gridSize,blockSize,0,get_cuda_stream()>>>(x,y,p,n);    
}
