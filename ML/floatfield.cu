////////////////////////////////////////////////////////////////////////
// field
// cuda doesn't have 4-dimensional kernel
// there is better ways to do it
// for the moment it is just done in 3d
// this experimental might not even work
////////////////////////////////////////////////////////////////////////


#include <cassert>
#include "cuda_runtime.h"
#include "math_constants.h"
#include <map>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "vector_float.h"
#include "field_float.h"


cudaStream_t get_cuda_stream();


/*
std::map<std::string,field_kernel1> field_map1;
std::map<std::string,field_kernel2> field_map2;
std::map<std::string,field_kernel3> field_map3;
std::map<std::string,field_kernel4> field_map4;

void register_field_kernel1(const char * name, field_kernel1 kernel) {
    // should assert if already exist?
    field_map1[name] = kernel;
}
void register_field_kernel2(const char * name, field_kernel2 kernel) {
    field_map2[name] = kernel;
}
void register_field_kernel3(const char * name, field_kernel3 kernel) {
    field_map3[name] = kernel;
}
void register_field_kernel4(const char * name, field_kernel4 kernel) {
    field_map4[name] = kernel;
}
float* execute_field_kernel1(const char * name, float * x, int M, int N, int O, int P) {
    typename std::map<std::string,field_kernel1>::iterator i = field_map1.find(name);
    // assert or return NULL?
    if(i == field_map1.end()) return NULL;
    return (i->second)(x,M,N,O*P);

}
float* execute_field_kernel2(const char * name, float * x, float * y, int M, int N, int O, int P) {
    typename std::map<std::string,field_kernel2>::iterator i = field_map2.find(name);
    // assert or return NULL?
    if(i == field_map2.end()) return NULL;
    return (i->second)(x,y,M,N,O*P);

}
float* execute_field_kernel3(const char * name, float * x, float * y, float * z, int M, int N, int O, int P) {
    typename std::map<std::string,field_kernel3>::iterator i = field_map3.find(name);
    // assert or return NULL?
    if(i == field_map3.end()) return NULL;
    return (i->second)(x,y,z,M,N,O*P);

}
float* execute_field_kernel4(const char * name, float * x, float * y, float * z, float * w, int M, int N, int O, int P) {
    typename std::map<std::string,field_kernel4>::iterator i = field_map4.find(name);
    // assert or return NULL?
    if(i == field_map4.end()) return NULL;
    return (i->second)(x,y,z,w,M,N,O*P);

}
*/



int dim(int M, int N, int O, int P) 
{
    return M*N*O*P;
}

__global__ void field_addf_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) 
        out[idx] = x[idx] + y[idx];
}

float* field_addf(float * x, float * y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));       
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_addf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}


void field_r_addf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_addf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
}



__global__ void field_subf_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)   out[idx] = x[idx] - y[idx];
}

float* field_subf(float * x, float * y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_subf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}


void field_r_subf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_subf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
}



__global__ void field_mulf_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)  out[idx] = x[idx] * y[idx];
}

float* field_mulf(float * x, float * y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_mulf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}

void field_r_mulf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_mulf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
}

__global__ void field_divf_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)  out[idx] = x[idx] / y[idx];
}

float* field_divf(float * x, float * y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_divf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}
void field_r_divf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_divf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
}

__global__ void field_modf_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)  out[idx] = fmodf(x[idx],y[idx]);
}

float* field_modf(float * x, float * y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_modf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}
void field_r_modf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_modf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
}



__global__ void field_acosf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)  out[idx] = acosf(in[idx]);
}

float* field_acosf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_acosf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);
    
    return output;
}
void field_r_acosf(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_acosf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);
        
}

__global__ void field_acoshf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = acoshf(in[idx]);
}

float* field_acoshf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));       
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_acoshf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}

void field_r_acoshf(float * devPtr, float * output, int M, int N, int O, int P)
{   
    int n = dim(M,N,O,P);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_acoshf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);
 
}

__global__ void field_asinhf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
            out[idx] = asinhf(in[idx]);
}

float* field_asinhf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_asinhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_asinhf(float * devPtr, float * output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        field_asinhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);
 
}


__global__ void field_asinf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = asinf(in[idx]);
}

float* field_asinf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_asinf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}

void field_r_asinf(float * devPtr, float * output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_asinf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}


__global__ void field_atan2f_device(float * a, float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = atan2f(a[idx],b[idx]);
}

float* field_atan2f(float * a, float * b, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_atan2f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O*P);

    return output;
}

void field_r_atan2f(float * a, float * b, float * output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_atan2f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O*P);
 
}

__global__ void field_atanf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = atanf(in[idx]);
}

float* field_atanf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_atanf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}

void field_r_atanf(float * devPtr, float * output, int M, int N, int O, int P)
{   
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_atanf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);
    
}



__global__ void field_atanhf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = atanhf(in[idx]);
}

float* field_atanhf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_atanhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}

void field_r_atanhf(float * devPtr, float * output, int M, int N, int O, int P)
{   
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_atanhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);
 
}



__global__ void field_ceilf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = ceilf(in[idx]);
}

float* field_ceilf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_ceilf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}

void field_r_ceilf(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_ceilf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}



__global__ void field_cosf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = cosf(in[idx]);
}

float* field_cosf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_cosf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_cosf(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_cosf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);
    
}


__global__ void field_coshf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = coshf(in[idx]);
}

float* field_coshf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_coshf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_coshf(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_coshf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}

__global__ void field_exp10f_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = exp10f(in[idx]);
}

float* field_exp10f(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_exp10f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_exp10f(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_exp10f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}


__global__ void field_exp2f_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = exp2f(in[idx]);
}

float* field_exp2f(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_exp2f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_exp2f(float * devPtr, float * output, int M, int N, int O, int P)
{   
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_exp2f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);
 
}



__global__ void field_expf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = expf(in[idx]);
}

float* field_expf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_expf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_expf(float * devPtr, float * output, int M, int N, int O, int P)
{   
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_expf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);
 
}

__global__ void field_expm1f_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = expm1f(in[idx]);
}

float* field_expm1f(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_expm1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_expm1f(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_expm1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}


__global__ void field_fabsf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fabsf(in[idx]);
}

float* field_fabsf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fabsf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_fabsf(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fabsf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}

__global__ void field_floorf_device(float * a,float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = floorf(a[idx]);
}

float* field_floorf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_floorf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_floorf(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_floorf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);
    
}



__global__ void field_fmaxf_device(float * a,float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fmaxf(a[idx],b[idx]);
}

float* field_fmaxf(float * x, float * y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fmaxf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

    return output;
}

void field_r_fmaxf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fmaxf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

}


__global__ void field_fminf_device(float * a,float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fminf(a[idx],b[idx]);
}

float* field_fminf(float * x, float * y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fminf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

    return output;
}

void field_r_fminf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fminf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

}

__global__ void field_fmodf_device(float * a,float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fmodf(a[idx],b[idx]);
}

float* field_fmodf(float * x, float * y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fmodf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

    return output;
}
void field_r_fmodf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fmodf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

}

__global__ void field_log10f_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = log10f(a[idx]);
}

float* field_log10f(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_log10f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_log10f(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_log10f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}


__global__ void field_log1pf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = log1pf(a[idx]);
}


float* field_log1pf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_log1pf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}

void field_r_log1pf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_log1pf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}


__global__ void field_log2f_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = log2f(a[idx]);
}

float* field_log2f(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_log2f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_log2f(float * x, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_log2f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}


__global__ void field_logbf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = logbf(a[idx]);
}

float* field_logbf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_logbf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_logbf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_logbf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}



__global__ void field_powf_device(float * a, float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = powf(a[idx],b[idx]);
}

float* field_powf(float * x, float * y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_powf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

    return output;
}
void field_r_powf(float * x, float * y, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_powf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

}




__global__ void field_rsqrtf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rsqrtf(a[idx]);
}

float* field_rsqrtf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_rsqrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}

void field_r_rsqrtf(float * x, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_rsqrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}



__global__ void field_sinf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = sinf(a[idx]);
}

float* field_sinf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sinf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_sinf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sinf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}


__global__ void field_sinhf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = sinhf(a[idx]);
}

float* field_sinhf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sinhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_sinhf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sinhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
    
}

__global__ void field_sqrtf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = sqrtf(a[idx]);
}

float* field_sqrtf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sqrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}

void field_r_sqrtf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sqrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}


__global__ void field_tanf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = tanf(a[idx]);
}

float* field_tanf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_tanf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_tanf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_tanf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}


__global__ void field_tanhf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = tanhf(a[idx]);
}

float* field_tanhf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_tanhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}

void field_r_tanhf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_tanhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}




__global__ void field_softmax_device(float * x,float *out, float sum, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = expf(x[idx]) / sum;
}

float* field_softmaxf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    float sum = vector_sumf(x,n);
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_softmax_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,sum,M,N,O*P);
        
    return output;
}
void field_r_softmaxf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float sum = vector_sumf(x,n);
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_softmax_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,sum,M,N,O*P);
            
}


__global__ void field_sigmoid_device(float * x, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = 1.0 / (1.0 + expf(-x[idx]));
}

float* field_sigmoidf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sigmoid_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
        
    return output;
}
void field_r_sigmoidf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sigmoid_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
        
}


__global__ void field_sigmoid_grad_device(float * x, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] * (1.0 - x[idx]);
}

float* field_sigmoid_gradf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sigmoid_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
        
    return output;
}


void field_r_sigmoid_gradf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sigmoid_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
        
}




__global__ void field_tanh_grad_device(float * x, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = 1.0 - (x[idx]*x[idx]);
}

float* field_tanh_gradf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_tanh_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
        
    return output;
}
void field_r_tanh_gradf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_tanh_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
        
}


__global__ void field_relu_device(float * x, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
    {
        if(x[idx] < 0) out[idx] = 0.0f;
        else out[idx] = x[idx];
    }
}

float* field_reluf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_relu_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
        
    return output;
}

void field_r_reluf(float * x, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_relu_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
        
}


__global__ void field_relu_grad_device(float * x, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
    {
        if(x[idx] > 0) out[idx] = 1.0;
        else out[idx] = 0.0f;
    }
}

float* field_relu_gradf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_relu_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
        
    return output;
}
void field_r_relu_gradf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_relu_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
        
}



__global__ void field_add_const_device(float * x, float y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] + y;
}

float* field_addf_const(float * x, float y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_add_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}

void field_r_addf_const(float * x, float y, float * output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_add_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
     
}


__global__ void field_sub_const_device(float * x, float y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] - y;
}

float* field_subf_const(float * x, float y, int M, int N, int O, int P)
{  
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sub_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}


void field_r_subf_const(float * x, float y, float *output, int M, int N, int O, int P)
{  
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sub_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
          
}


__global__ void field_mul_const_device(float * x, float y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] * y;
}

float* field_mulf_const(float * x, float y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_mul_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
     
    return output;
}
void field_r_mulf_const(float * x, float y, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_mul_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}

__global__ void field_div_const_device(float * x, float y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] / y;
}

float* field_divf_const(float * x, float y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_div_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_divf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_div_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}


__global__ void field_mod_const_device(float * x, float y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = fmodf(x[idx],y);
}

float* field_modf_const(float * x, float y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_mod_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}

void field_r_modf_const(float * x, float y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_mod_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}



__global__ void field_atan2f_const_device(float * a, float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = atan2f(a[idx],b);
}

float* field_atan2f_const(float * a, float  b, int M, int N, int O, int P)
{   
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_atan2f_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O*P);
    
    return output;
}


void field_r_atan2f_const(float * a, float  b, float * output, int M, int N, int O, int P)
{   
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_atan2f_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O*P);
    
}





__global__ void field_fmaxf_const_device(float * a,float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fmaxf(a[idx],b);
}

float* field_fmaxf_const(float * x, float y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fmaxf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}

void field_r_fmaxf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fmaxf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}


__global__ void field_fminf_const_device(float * a,float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fminf(a[idx],b);
}

float* field_fminf_const(float * x, float y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fminf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}
void field_r_fminf_const(float * x, float y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fminf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}

__global__ void field_fmodf_const_device(float * a,float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fmodf(a[idx],b);
}

float* field_fmodf_const(float * x, float y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_fmodf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}
void field_r_fmodf_const(float * x, float y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fmodf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
       
}



__global__ void field_powf_const_device(float * a, float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = powf(a[idx],b);
}

float* field_powf_const(float * x, float y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_powf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}
void field_r_powf_const(float * x, float y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_powf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
}






/////////////////////////////////
// const/scalar
/////////////////////////////////
__global__ void field_add_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] + y[0];
}



float* field_addf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_add_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_addf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{      
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_add_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
            
}




__global__ void field_sub_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] - y[0];
}

float* field_subf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        field_sub_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_subf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_sub_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}


__global__ void field_mul_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] * y[0];
}



float* field_mulf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_mul_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_mulf_scalar(float * x, float * y, float * output, int M, int N, int O, int P)
{        
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_mul_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
            
}


__global__ void field_div_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O && y[0] != 0.0f) out[idx] = x[idx] / y[0];
}

float* field_divf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_div_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_divf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_div_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}



__global__ void field_mod_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = fmodf(x[idx],y[0]);
}

float* field_modf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_mod_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_modf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_mod_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}


float* field_fmodf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_mod_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_fmodf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_mod_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}

__global__ void field_fmaxf_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = fmaxf(x[idx],y[0]);
}

float* field_fmaxf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_fmaxf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_fmaxf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fmaxf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
}

__global__ void field_fminf_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = fminf(x[idx],y[0]);
}

float* field_fminf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_fminf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_fminf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fminf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}


__global__ void field_pow_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = powf(x[idx],y[0]);
}

float* field_powf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_pow_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_powf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_pow_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}

__global__ void field_atan2_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = atan2f(x[idx],y[0]);
}

float* field_atan2f_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_atan2_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_atan2f_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_atan2_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}

__global__ void field_fdim_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = fdimf(x[idx],y[0]);
}

float* field_fdimf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_fdim_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_fdimf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fdim_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);        
}

__global__ void field_fdivide_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = fdividef(x[idx],y[0]);
}

float* field_fdividef_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_fdivide_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_fdividef_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_fdivide_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}

__global__ void field_remainder_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = remainderf(x[idx],y[0]);
}

float* field_remainderf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_remainder_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_remainderf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_remainder_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}


__global__ void field_hypot_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = hypotf(x[idx],y[0]);
}

float* field_hypotf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_hypot_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_hypotf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_hypot_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);        
}


__global__ void field_rhypot_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = rhypotf(x[idx],y[0]);
}

float* field_rhypotf_scalar(float * x, float * y, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_rhypot_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
    return output;
}
void field_r_rhypotf_scalar(float * x, float * y, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_rhypot_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);        
}

// vector.cu
__global__ void field_cbrtf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = atanf(in[idx]);
}

float* field_cbrtf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_cbrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_cbrtf(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    field_cbrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}





__global__ void field_copysignf_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = copysignf(x[idx],y[idx]);
}

float* field_copysignf(float * X, float *Y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_copysignf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(X,Y,output,M,N,O*P);

    return output;
}

void field_r_copysignf(float * X, float *Y, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    field_copysignf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(X,Y,output,M,N,O*P);

}

__global__ void field_cospif_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = cospif(in[idx]);
}

float* field_cospif(float * devPtr, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_cospif_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_cospif(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_cospif_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}

__global__ void field_cyl_bessel_i0f_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = cyl_bessel_i0f(in[idx]);
}

float* field_cyl_bessel_i0f(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_cyl_bessel_i0f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_cyl_bessel_i0f(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_cyl_bessel_i0f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}

__global__ void field_cyl_bessel_i1f_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = cyl_bessel_i1f(in[idx]);
}

float* field_cyl_bessel_i1f(float * devPtr, int M, int N, int O, int P)
{
   int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_cyl_bessel_i1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_cyl_bessel_i1f(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_cyl_bessel_i1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}

__global__ void field_erfcf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = erfcf(in[idx]);
}

float* field_erfcf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_erfcf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_erfcf(float * devPtr, float * output, int M, int N, int O, int P)
{ 
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_erfcf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}


__global__ void field_erfcinvf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = erfcinvf(in[idx]);
}

float* field_erfcinvf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_erfcinvf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_erfcinvf(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
 
    field_erfcinvf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}



__global__ void field_erfcxf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = erfcxf(in[idx]);
}

float* field_erfcxf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
     field_erfcxf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_erfcxf(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_erfcxf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}

__global__ void field_erff_device(float * in, float * out, int M, int N, int O)
{
     int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)

        out[idx] = erff(in[idx]);
}

float* field_erff(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_erff_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_erff(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_erff_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}


__global__ void field_erfinvf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = erfinvf(in[idx]);
}

float* field_erfinvf(float * devPtr, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_erfinvf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

    return output;
}
void field_r_erfinvf(float * devPtr, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_erfinvf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O*P);

}

__global__ void field_fdimf_device(float * a, float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fdimf(a[idx],b[idx]);
}

float* field_fdimf(float * a, float * b, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_fdimf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O*P);

    return output;
}
void field_r_fdimf(float * a, float * b, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_fdimf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O*P);

}

__global__ void field_fdividef_device(float * a, float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fdividef(a[idx],b[idx]);
}

float* field_fdividef(float * a, float * b, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_fdividef_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O*P);

    return output;
}
void field_r_fdividef(float * a, float * b, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_fdividef_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O*P);

}

__global__ void field_fmaf_device(float * a, float * b, float * c, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fmaf(a[idx],b[idx],c[idx]);
}

float* field_fmaf(float * x, float * y, float * z, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_fmaf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N,O*P);

    return output;
}
void field_r_fmaf(float * x, float * y, float * z, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_fmaf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N,O*P);

}

__global__ void field_hypotf_device(float * a,float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = hypotf(a[idx],b[idx]);
}

float* field_hypotf(float * x, float * y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_hypotf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

    return output;
}
void field_r_hypotf(float * x, float * y, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_hypotf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

}

__global__ void field_ilogbf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = ilogbf(a[idx]);
}

float* field_ilogbf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_ilogbf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_ilogbf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_ilogbf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}


__global__ void field_j0f_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = j0f(a[idx]);
}

float* field_j0f(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_j0f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_j0f(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_j0f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}

__global__ void field_j1f_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = j1f(a[idx]);
}

float* field_j1f(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_j1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_j1f(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_j1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}

__global__ void field_jnf_device(float * a, int m, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = jnf(a[idx],m);
}

float* field_jnf(float * x, int m, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_jnf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,m,output,M,N,O*P);

    return output;
}
void field_r_jnf(float * x, float * output, int m, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_jnf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,m,output,M,N,O*P);

}


__global__ void field_ldexpf_device(float * a, int exp, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = ldexpf(a[idx],exp);
}

float* field_ldexpf(float * x, int exp, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_ldexpf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,exp,output,M,N,O*P);

    return output;
}
void field_r_ldexpf(float * x, float * output, int exp, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_ldexpf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,exp,output,M,N,O*P);

}


__global__ void field_lgammaf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = lgammaf(a[idx]);
}

float* field_lgammaf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_lgammaf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_lgammaf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_lgammaf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}


__global__ void field_nearbyintf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = lroundf(a[idx]);
}

float* field_nearbyintf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_nearbyintf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_nearbyintf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            

    field_nearbyintf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}

__global__ void field_norm3df_device(float * a, float * b, float * c, float* out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = norm3df(a[idx],b[idx],c[idx]);
}

float* field_norm3df(float * x, float * y, float * z, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_norm3df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N,O*P);

    return output;
}
void field_r_norm3df(float * x, float * y, float * z, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            

    field_norm3df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N,O*P);

}


__global__ void field_norm4df_device(float * a, float * b, float * c, float * d, float* out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = norm4df(a[idx],b[idx],c[idx],d[idx]);
}

float* field_norm4df(float * x, float * y, float * z, float * q, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            

    field_norm4df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,q,output,M,N,O*P);

    return output;
}
void field_r_norm4df(float * x, float * y, float * z, float * q, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_norm4df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,q,output,M,N,O*P);

}


__global__ void field_normcdff_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = normcdff(a[idx]);
}

float* field_normcdff(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_normcdff_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_normcdff(float * x, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_normcdff_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
    
}


__global__ void field_normcdfinvf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = normcdfinvf(a[idx]);
}

float* field_normcdfinvf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_normcdfinvf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_normcdfinvf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_normcdfinvf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}



__global__ void field_normf_device(int dim, const float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = normf(dim,a);
}

float* field_normf(int d, float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);                
    field_normf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(d,x,output,M,N,O*P);

    return output;
}
void field_r_normf(int d, float * x, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_normf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(d,x,output,M,N,O*P);

}

__global__ void field_rcbrtf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rcbrtf(a[idx]);
}

float* field_rcbrtf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_rcbrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_rcbrtf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_rcbrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}


__global__ void field_remainderf_device(float * a, float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = remainderf(a[idx],b[idx]);
}


float* field_remainderf(float * x, float * y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_remainderf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

    return output;
}
void field_r_remainderf(float * x, float * y, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_remainderf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

}

__global__ void field_rhypotf_device(float * a, float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rhypotf(a[idx],b[idx]);
}

float* field_rhypotf(float * x, float * y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_rhypotf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);

    return output;
}
void field_r_rhypotf(float * x, float * y, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_rhypotf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
 
}




__global__ void field_rnorm3df_device(float * a, float * b, float * c, float* out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rnorm3df(a[idx],b[idx],c[idx]);
}

float* field_rnorm3df(float * x, float * y, float * z, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_rnorm3df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N,O*P);

    return output;
}
void field_r_rnorm3df(float * x, float * y, float * z, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_rnorm3df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N,O*P);

}

__global__ void field_rnorm4df_device(float * a, float * b, float * c, float * d, float* out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rnorm4df(a[idx],b[idx],c[idx],d[idx]);
}


float* field_rnorm4df(float * x, float * y, float * z, float * q, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_rnorm4df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,q,output,M,N,O*P);

    return output;
}
void field_r_rnorm4df(float * x, float * y, float * z, float * q, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_rnorm4df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,q,output,M,N,O*P);

}


__global__ void field_rnormf_device(int dim, const float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rnormf(dim,a);
}

float* field_rnormf(int d, float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);                
    field_rnormf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(d,x,output,M,N,O*P);

    return output;
}
void field_r_rnormf(int d, float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_rnormf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(d,x,output,M,N,O*P);

}

__global__ void field_scalblnf_device(float * a, long int m,float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = scalblnf(a[idx],m);
}


float* field_scalblnf(float * x, long int m, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_scalblnf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,m,output,M,N,O*P);

    return output;
}
void field_r_scalblnf(float * x, long int m, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_scalblnf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,m,output,M,N,O*P);

}

__global__ void field_sinpif_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = sinpif(a[idx]);
}

float* field_sinpif(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);                
    field_sinpif_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_sinpif(float * x, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_sinpif_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);
 
}

__global__ void field_tgammaf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = tgammaf(a[idx]);
}

float* field_tgammaf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_tgammaf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_tgammaf(float * x, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_tgammaf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}


__global__ void field_truncf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = truncf(a[idx]);
}

float* field_truncf(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);                field_truncf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_truncf(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_truncf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}

__global__ void field_y0f_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = y0f(a[idx]);
}

float* field_y0f(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_y0f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_y0f(float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_y0f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}



__global__ void field_y1f_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = y1f(a[idx]);
}

float* field_y1f(float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_y1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

    return output;
}
void field_r_y1f(float * x, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_y1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O*P);

}


__global__ void field_ynf_device(int m, float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = ynf(m,a[idx]);
}

float* field_ynf(int m, float * x, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_ynf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(m,x,output,M,N,O*P);

    return output;
}
void field_r_ynf(int m, float * x, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_ynf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(m,x,output,M,N,O*P);

}

__global__ void field_fdimf_const_device(float * a, float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fdimf(a[idx],b);
}

float* field_fdimf_const(float * a, float  b, int M, int N, int O, int P)
{
int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_fdimf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O*P);
    
    return output;
}
void field_r_fdimf_const(float * a, float  b, float * output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_fdimf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O*P);
        
}


__global__ void field_fdividef_const_device(float * a, float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fdividef(a[idx],b);
}

float* field_fdividef_const(float * a, float b, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_fdividef_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O*P);
    
    return output;
}
void field_r_fdividef_const(float * a, float b, float *output, int M, int N, int O, int P)
{    
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_fdividef_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O*P);
     
}

__global__ void field_hypotf_const_device(float * a,float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = hypotf(a[idx],b);
}

float* field_hypotf_const(float * x, float y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_hypotf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}
void field_r_hypotf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_hypotf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
}



__global__ void field_remainderf_const_device(float * a, float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = remainderf(a[idx],b);
}


float* field_remainderf_const(float * x, float y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_remainderf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}
void field_r_remainderf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_remainderf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
        
}

__global__ void field_rhypotf_const_device(float * a, float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rhypotf(a[idx],b);
}

float* field_rhypotf_const(float * x, float y, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_rhypotf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
    return output;
}
void field_r_rhypotf_const(float * x, float y, float *output, int M, int N, int O, int P)
{
    int n = dim(M,N,O,P);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O*P) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    field_rhypotf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O*P);
    
}
