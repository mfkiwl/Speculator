////////////////////////////////////////////////////////////////////////
// cube
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
#include "cube_float.h"


cudaStream_t get_cuda_stream();


/*
std::map<std::string,cube_kernel1> cube_map1;
std::map<std::string,cube_kernel2> cube_map2;
std::map<std::string,cube_kernel3> cube_map3;
std::map<std::string,cube_kernel4> cube_map4;

void register_cube_kernel1(const char * name, cube_kernel1 kernel) {
    // should assert if already exist?
    cube_map1[name] = kernel;
}
void register_cube_kernel2(const char * name, cube_kernel2 kernel) {
    cube_map2[name] = kernel;
}
void register_cube_kernel3(const char * name, cube_kernel3 kernel) {
    cube_map3[name] = kernel;
}
void register_cube_kernel4(const char * name, cube_kernel4 kernel) {
    cube_map4[name] = kernel;
}
float* execute_cube_kernel1(const char * name, float * x, int M, int N, int O) {
    typename std::map<std::string,cube_kernel1>::iterator i = cube_map1.find(name);
    // assert or return NULL?
    if(i == cube_map1.end()) return NULL;
    return (i->second)(x,M,N,O);
}
float* execute_cube_kernel2(const char * name, float * x, float * y, int M, int N, int O) {
    typename std::map<std::string,cube_kernel2>::iterator i = cube_map2.find(name);
    // assert or return NULL?
    if(i == cube_map2.end()) return NULL;
    return (i->second)(x,y,M,N,O);
}
float* execute_cube_kernel3(const char * name, float * x, float * y, float * z, int M, int N, int O) {
    typename std::map<std::string,cube_kernel3>::iterator i = cube_map3.find(name);
    // assert or return NULL?
    if(i == cube_map3.end()) return NULL;
    return (i->second)(x,y,z,M,N,O);
}
float* execute_cube_kernel4(const char * name, float * x, float * y, float * z, float * w, int M, int N, int O) {
    typename std::map<std::string,cube_kernel4>::iterator i = cube_map4.find(name);
    // assert or return NULL?
    if(i == cube_map4.end()) return NULL;
    return (i->second)(x,y,z,w,M,N,O);
}
*/



int dim(int M, int N, int O) 
{
    return M*N*O;
}

__global__ void cube_addf_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) 
        out[idx] = x[idx] + y[idx];
}

float* cube_addf(float * x, float * y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));       
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_addf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}


void cube_r_addf(float * x, float * y, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_addf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
}



__global__ void cube_subf_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)   out[idx] = x[idx] - y[idx];
}

float* cube_subf(float * x, float * y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_subf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}


void cube_r_subf(float * x, float * y, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_subf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
}



__global__ void cube_mulf_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)  out[idx] = x[idx] * y[idx];
}

float* cube_mulf(float * x, float * y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_mulf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}

void cube_r_mulf(float * x, float * y, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_mulf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
}

__global__ void cube_divf_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)  out[idx] = x[idx] / y[idx];
}

float* cube_divf(float * x, float * y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_divf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}
void cube_r_divf(float * x, float * y, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_divf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
}

__global__ void cube_modf_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)  out[idx] = fmodf(x[idx],y[idx]);
}

float* cube_modf(float * x, float * y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_modf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}
void cube_r_modf(float * x, float * y, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_modf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
}



__global__ void cube_acosf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)  out[idx] = acosf(in[idx]);
}

float* cube_acosf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_acosf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);    
    return output;
}
void cube_r_acosf(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_acosf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);        
}

__global__ void cube_acoshf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = acoshf(in[idx]);
}

float* cube_acoshf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));       
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_acoshf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}

void cube_r_acoshf(float * devPtr, float * output, int M, int N, int O)
{   
    int n = dim(M,N,O);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_acoshf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O); 
}

__global__ void cube_asinhf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
            out[idx] = asinhf(in[idx]);
}

float* cube_asinhf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_asinhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_asinhf(float * devPtr, float * output, int M, int N, int O)
{    
    int n = dim(M,N,O);        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        cube_asinhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O); 
}


__global__ void cube_asinf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = asinf(in[idx]);
}

float* cube_asinf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_asinf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}

void cube_r_asinf(float * devPtr, float * output, int M, int N, int O)
{    
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_asinf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}


__global__ void cube_atan2f_device(float * a, float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = atan2f(a[idx],b[idx]);
}

float* cube_atan2f(float * a, float * b, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_atan2f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O);
    return output;
}

void cube_r_atan2f(float * a, float * b, float * output, int M, int N, int O)
{    
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_atan2f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O); 
}

__global__ void cube_atanf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = atanf(in[idx]);
}

float* cube_atanf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_atanf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}

void cube_r_atanf(float * devPtr, float * output, int M, int N, int O)
{   
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_atanf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);    
}



__global__ void cube_atanhf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = atanhf(in[idx]);
}

float* cube_atanhf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_atanhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}

void cube_r_atanhf(float * devPtr, float * output, int M, int N, int O)
{   
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_atanhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O); 
}



__global__ void cube_ceilf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = ceilf(in[idx]);
}

float* cube_ceilf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_ceilf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}

void cube_r_ceilf(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_ceilf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}



__global__ void cube_cosf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = cosf(in[idx]);
}

float* cube_cosf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_cosf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_cosf(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_cosf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);    
}


__global__ void cube_coshf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = coshf(in[idx]);
}

float* cube_coshf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_coshf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_coshf(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_coshf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}

__global__ void cube_exp10f_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = exp10f(in[idx]);
}

float* cube_exp10f(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_exp10f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_exp10f(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_exp10f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}


__global__ void cube_exp2f_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = exp2f(in[idx]);
}

float* cube_exp2f(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_exp2f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_exp2f(float * devPtr, float * output, int M, int N, int O)
{   
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_exp2f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O); 
}



__global__ void cube_expf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = expf(in[idx]);
}

float* cube_expf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_expf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_expf(float * devPtr, float * output, int M, int N, int O)
{   
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_expf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O); 
}

__global__ void cube_expm1f_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = expm1f(in[idx]);
}

float* cube_expm1f(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_expm1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_expm1f(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_expm1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}


__global__ void cube_fabsf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fabsf(in[idx]);
}

float* cube_fabsf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fabsf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_fabsf(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fabsf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}

__global__ void cube_floorf_device(float * a,float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = floorf(a[idx]);
}

float* cube_floorf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_floorf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_floorf(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_floorf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);    
}



__global__ void cube_fmaxf_device(float * a,float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fmaxf(a[idx],b[idx]);
}

float* cube_fmaxf(float * x, float * y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fmaxf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
    return output;
}

void cube_r_fmaxf(float * x, float * y, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fmaxf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
}


__global__ void cube_fminf_device(float * a,float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fminf(a[idx],b[idx]);
}

float* cube_fminf(float * x, float * y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fminf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
    return output;
}

void cube_r_fminf(float * x, float * y, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fminf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
}

__global__ void cube_fmodf_device(float * a,float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fmodf(a[idx],b[idx]);
}

float* cube_fmodf(float * x, float * y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fmodf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
    return output;
}
void cube_r_fmodf(float * x, float * y, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fmodf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
}

__global__ void cube_log10f_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = log10f(a[idx]);
}

float* cube_log10f(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_log10f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_log10f(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_log10f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}


__global__ void cube_log1pf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = log1pf(a[idx]);
}


float* cube_log1pf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_log1pf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}

void cube_r_log1pf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_log1pf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}


__global__ void cube_log2f_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = log2f(a[idx]);
}

float* cube_log2f(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_log2f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_log2f(float * x, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_log2f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}


__global__ void cube_logbf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = logbf(a[idx]);
}

float* cube_logbf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_logbf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_logbf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_logbf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}



__global__ void cube_powf_device(float * a, float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = powf(a[idx],b[idx]);
}

float* cube_powf(float * x, float * y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_powf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
    return output;
}
void cube_r_powf(float * x, float * y, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_powf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
}




__global__ void cube_rsqrtf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rsqrtf(a[idx]);
}

float* cube_rsqrtf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_rsqrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}

void cube_r_rsqrtf(float * x, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_rsqrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}



__global__ void cube_sinf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = sinf(a[idx]);
}

float* cube_sinf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sinf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_sinf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sinf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}


__global__ void cube_sinhf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = sinhf(a[idx]);
}

float* cube_sinhf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sinhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_sinhf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sinhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);    
}

__global__ void cube_sqrtf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = sqrtf(a[idx]);
}

float* cube_sqrtf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sqrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}

void cube_r_sqrtf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sqrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}


__global__ void cube_tanf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = tanf(a[idx]);
}

float* cube_tanf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_tanf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_tanf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_tanf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}


__global__ void cube_tanhf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = tanhf(a[idx]);
}

float* cube_tanhf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_tanhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}

void cube_r_tanhf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_tanhf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}




__global__ void cube_softmax_device(float * x,float *out, float sum, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = expf(x[idx]) / sum;
}

float* cube_softmaxf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    float sum = vector_sumf(x,n);
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_softmax_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,sum,M,N,O);        
    return output;
}
void cube_r_softmaxf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    float sum = vector_sumf(x,n);
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_softmax_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,sum,M,N,O);            
}


__global__ void cube_sigmoid_device(float * x, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = 1.0 / (1.0 + expf(-x[idx]));
}

float* cube_sigmoidf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sigmoid_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);        
    return output;
}
void cube_r_sigmoidf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sigmoid_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);        
}


__global__ void cube_sigmoid_grad_device(float * x, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] * (1.0 - x[idx]);
}

float* cube_sigmoid_gradf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sigmoid_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);        
    return output;
}


void cube_r_sigmoid_gradf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sigmoid_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);        
}




__global__ void cube_tanh_grad_device(float * x, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = 1.0 - (x[idx]*x[idx]);
}

float* cube_tanh_gradf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_tanh_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);        
    return output;
}
void cube_r_tanh_gradf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_tanh_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);        
}


__global__ void cube_relu_device(float * x, float * out, int M, int N, int O)
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

float* cube_reluf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_relu_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);        
    return output;
}

void cube_r_reluf(float * x, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_relu_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);        
}


__global__ void cube_relu_grad_device(float * x, float * out, int M, int N, int O)
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

float* cube_relu_gradf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_relu_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);        
    return output;
}
void cube_r_relu_gradf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_relu_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);        
}



__global__ void cube_add_const_device(float * x, float y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] + y;
}

float* cube_addf_const(float * x, float y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_add_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}

void cube_r_addf_const(float * x, float y, float * output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_add_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);     
}


__global__ void cube_sub_const_device(float * x, float y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] - y;
}

float* cube_subf_const(float * x, float y, int M, int N, int O)
{  
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sub_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
    return output;
}


void cube_r_subf_const(float * x, float y, float *output, int M, int N, int O)
{  
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sub_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);          
}


__global__ void cube_mul_const_device(float * x, float y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] * y;
}

float* cube_mulf_const(float * x, float y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_mul_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);     
    return output;
}
void cube_r_mulf_const(float * x, float y, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_mul_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}

__global__ void cube_div_const_device(float * x, float y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] / y;
}

float* cube_divf_const(float * x, float y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_div_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
    return output;
}
void cube_r_divf_const(float * x, float y, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_div_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}


__global__ void cube_mod_const_device(float * x, float y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = fmodf(x[idx],y);
}

float* cube_modf_const(float * x, float y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_mod_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
    return output;
}

void cube_r_modf_const(float * x, float y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_mod_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}



__global__ void cube_atan2f_const_device(float * a, float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = atan2f(a[idx],b);
}

float* cube_atan2f_const(float * a, float  b, int M, int N, int O)
{   
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_atan2f_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O);    
    return output;
}


void cube_r_atan2f_const(float * a, float  b, float * output, int M, int N, int O)
{   
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_atan2f_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O);    
}





__global__ void cube_fmaxf_const_device(float * a,float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fmaxf(a[idx],b);
}

float* cube_fmaxf_const(float * x, float y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fmaxf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}

void cube_r_fmaxf_const(float * x, float y, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fmaxf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}


__global__ void cube_fminf_const_device(float * a,float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fminf(a[idx],b);
}

float* cube_fminf_const(float * x, float y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fminf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}
void cube_r_fminf_const(float * x, float y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fminf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}

__global__ void cube_fmodf_const_device(float * a,float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fmodf(a[idx],b);
}

float* cube_fmodf_const(float * x, float y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_fmodf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}
void cube_r_fmodf_const(float * x, float y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fmodf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);       
}



__global__ void cube_powf_const_device(float * a, float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = powf(a[idx],b);
}

float* cube_powf_const(float * x, float y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_powf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}
void cube_r_powf_const(float * x, float y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_powf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
}






/////////////////////////////////
// const/scalar
/////////////////////////////////
__global__ void cube_add_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] + y[0];
}



float* cube_addf_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_add_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
    return output;
}
void cube_r_addf_scalar(float * x, float * y, float *output, int M, int N, int O)
{      
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_add_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);            
}




__global__ void cube_sub_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] - y[0];
}

float* cube_subf_scalar(float * x, float * y, int M, int N, int O)
{    
int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        cube_sub_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
    return output;
}
void cube_r_subf_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_sub_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}


__global__ void cube_mul_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = x[idx] * y[0];
}



float* cube_mulf_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_mul_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
    return output;
}
void cube_r_mulf_scalar(float * x, float * y, float * output, int M, int N, int O)
{        
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_mul_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);            
}


__global__ void cube_div_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O && y[0] != 0.0f) out[idx] = x[idx] / y[0];
}

float* cube_divf_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_div_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
    return output;
}
void cube_r_divf_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_div_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}



__global__ void cube_mod_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = fmodf(x[idx],y[0]);
}

float* cube_modf_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_mod_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
    return output;
}
void cube_r_modf_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_mod_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}


float* cube_fmodf_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_mod_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
    return output;
}
void cube_r_fmodf_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_mod_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}



__global__ void cube_fmaxf_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = fmaxf(x[idx],y[0]);
}

float* cube_fmaxf_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_fmaxf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
        
    return output;
}
void cube_r_fmaxf_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fmaxf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
}

__global__ void cube_fminf_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = fminf(x[idx],y[0]);
}

float* cube_fminf_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_fminf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
        
    return output;
}
void cube_r_fminf_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fminf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
        
}


__global__ void cube_pow_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = powf(x[idx],y[0]);
}

float* cube_powf_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_pow_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
        
    return output;
}
void cube_r_powf_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_pow_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
        
}

__global__ void cube_atan2_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = atan2f(x[idx],y[0]);
}

float* cube_atan2f_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_atan2_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
    return output;
}
void cube_r_atan2f_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_atan2_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}

__global__ void cube_fdim_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = fdimf(x[idx],y[0]);
}


float* cube_fdimf_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_fdim_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
        
    return output;
}
void cube_r_fdimf_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fdim_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
        
}

__global__ void cube_fdivide_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = fdividef(x[idx],y[0]);
}

float* cube_fdividef_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_fdivide_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
        
    return output;
}
void cube_r_fdividef_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_fdivide_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
        
}

__global__ void cube_remainder_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = remainderf(x[idx],y[0]);
}

float* cube_remainderf_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_remainder_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
        
    return output;
}
void cube_r_remainderf_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_remainder_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
        
}


__global__ void cube_hypot_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = hypotf(x[idx],y[0]);
}

float* cube_hypotf_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_hypot_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
        
    return output;
}
void cube_r_hypotf_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_hypot_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}

__global__ void cube_rhypot_scalar_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O) out[idx] = rhypotf(x[idx],y[0]);
}

float* cube_rhypotf_scalar(float * x, float * y, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_rhypot_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
    return output;
}
void cube_r_rhypotf_scalar(float * x, float * y, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = ((O) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_rhypot_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}


// vector.cu
__global__ void cube_cbrtf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = atanf(in[idx]);
}

float* cube_cbrtf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_cbrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_cbrtf(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);    
    cube_cbrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}


__global__ void cube_copysignf_device(float * x, float * y, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = copysignf(x[idx],y[idx]);
}

float* cube_copysignf(float * X, float *Y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_copysignf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(X,Y,output,M,N,O);
    return output;
}

void cube_r_copysignf(float * X, float *Y, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);        
    cube_copysignf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(X,Y,output,M,N,O);
}

__global__ void cube_cospif_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = cospif(in[idx]);
}

float* cube_cospif(float * devPtr, int M, int N, int O)
{    
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_cospif_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_cospif(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_cospif_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}

__global__ void cube_cyl_bessel_i0f_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = cyl_bessel_i0f(in[idx]);
}

float* cube_cyl_bessel_i0f(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_cyl_bessel_i0f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_cyl_bessel_i0f(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_cyl_bessel_i0f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}

__global__ void cube_cyl_bessel_i1f_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = cyl_bessel_i1f(in[idx]);
}

float* cube_cyl_bessel_i1f(float * devPtr, int M, int N, int O)
{
   int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_cyl_bessel_i1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_cyl_bessel_i1f(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_cyl_bessel_i1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}

__global__ void cube_erfcf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = erfcf(in[idx]);
}

float* cube_erfcf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_erfcf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_erfcf(float * devPtr, float * output, int M, int N, int O)
{ 
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_erfcf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}


__global__ void cube_erfcinvf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = erfcinvf(in[idx]);
}

float* cube_erfcinvf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_erfcinvf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_erfcinvf(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
 
    cube_erfcinvf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}



__global__ void cube_erfcxf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = erfcxf(in[idx]);
}

float* cube_erfcxf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
     cube_erfcxf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_erfcxf(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_erfcxf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}

__global__ void cube_erff_device(float * in, float * out, int M, int N, int O)
{
     int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)

        out[idx] = erff(in[idx]);
}

float* cube_erff(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_erff_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_erff(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_erff_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}


__global__ void cube_erfinvf_device(float * in, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = erfinvf(in[idx]);
}

float* cube_erfinvf(float * devPtr, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_erfinvf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
    return output;
}
void cube_r_erfinvf(float * devPtr, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_erfinvf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(devPtr,output,M,N,O);
}

__global__ void cube_fdimf_device(float * a, float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fdimf(a[idx],b[idx]);
}

float* cube_fdimf(float * a, float * b, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_fdimf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O);
    return output;
}
void cube_r_fdimf(float * a, float * b, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_fdimf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O);
}

__global__ void cube_fdividef_device(float * a, float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fdividef(a[idx],b[idx]);
}

float* cube_fdividef(float * a, float * b, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_fdividef_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O);
    return output;
}
void cube_r_fdividef(float * a, float * b, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_fdividef_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O);
}

__global__ void cube_fmaf_device(float * a, float * b, float * c, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fmaf(a[idx],b[idx],c[idx]);
}

float* cube_fmaf(float * x, float * y, float * z, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_fmaf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N,O);
    return output;
}
void cube_r_fmaf(float * x, float * y, float * z, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_fmaf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N,O);
}

__global__ void cube_hypotf_device(float * a,float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = hypotf(a[idx],b[idx]);
}

float* cube_hypotf(float * x, float * y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_hypotf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
    return output;
}
void cube_r_hypotf(float * x, float * y, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_hypotf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
}

__global__ void cube_ilogbf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = ilogbf(a[idx]);
}

float* cube_ilogbf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_ilogbf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_ilogbf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_ilogbf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}


__global__ void cube_j0f_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = j0f(a[idx]);
}

float* cube_j0f(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_j0f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_j0f(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_j0f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}

__global__ void cube_j1f_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = j1f(a[idx]);
}

float* cube_j1f(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_j1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_j1f(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_j1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}

__global__ void cube_jnf_device(float * a, int m, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = jnf(a[idx],m);
}

float* cube_jnf(float * x, int m, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_jnf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,m,output,M,N,O);
    return output;
}
void cube_r_jnf(float * x, float * output, int m, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_jnf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,m,output,M,N,O);
}


__global__ void cube_ldexpf_device(float * a, int exp, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = ldexpf(a[idx],exp);
}

float* cube_ldexpf(float * x, int exp, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_ldexpf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,exp,output,M,N,O);
    return output;
}
void cube_r_ldexpf(float * x, float * output, int exp, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_ldexpf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,exp,output,M,N,O);
}


__global__ void cube_lgammaf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = lgammaf(a[idx]);
}

float* cube_lgammaf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_lgammaf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_lgammaf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_lgammaf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}


__global__ void cube_nearbyintf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = lroundf(a[idx]);
}

float* cube_nearbyintf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_nearbyintf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_nearbyintf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            

    cube_nearbyintf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}

__global__ void cube_norm3df_device(float * a, float * b, float * c, float* out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = norm3df(a[idx],b[idx],c[idx]);
}

float* cube_norm3df(float * x, float * y, float * z, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_norm3df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N,O);
    return output;
}
void cube_r_norm3df(float * x, float * y, float * z, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            

    cube_norm3df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N,O);
}


__global__ void cube_norm4df_device(float * a, float * b, float * c, float * d, float* out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = norm4df(a[idx],b[idx],c[idx],d[idx]);
}

float* cube_norm4df(float * x, float * y, float * z, float * q, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            

    cube_norm4df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,q,output,M,N,O);
    return output;
}
void cube_r_norm4df(float * x, float * y, float * z, float * q, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_norm4df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,q,output,M,N,O);
}


__global__ void cube_normcdff_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = normcdff(a[idx]);
}

float* cube_normcdff(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_normcdff_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_normcdff(float * x, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_normcdff_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);    
}


__global__ void cube_normcdfinvf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = normcdfinvf(a[idx]);
}

float* cube_normcdfinvf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_normcdfinvf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_normcdfinvf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_normcdfinvf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}



__global__ void cube_normf_device(int dim, const float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = normf(dim,a);
}

float* cube_normf(int d, float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));        
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);                
    cube_normf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(d,x,output,M,N,O);
    return output;
}
void cube_r_normf(int d, float * x, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_normf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(d,x,output,M,N,O);
}

__global__ void cube_rcbrtf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rcbrtf(a[idx]);
}

float* cube_rcbrtf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_rcbrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_rcbrtf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_rcbrtf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}


__global__ void cube_remainderf_device(float * a, float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = remainderf(a[idx],b[idx]);
}


float* cube_remainderf(float * x, float * y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_remainderf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
    return output;
}
void cube_r_remainderf(float * x, float * y, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_remainderf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
}

__global__ void cube_rhypotf_device(float * a, float * b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rhypotf(a[idx],b[idx]);
}

float* cube_rhypotf(float * x, float * y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_rhypotf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);
    return output;
}
void cube_r_rhypotf(float * x, float * y, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_rhypotf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O); 
}




__global__ void cube_rnorm3df_device(float * a, float * b, float * c, float* out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rnorm3df(a[idx],b[idx],c[idx]);
}

float* cube_rnorm3df(float * x, float * y, float * z, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_rnorm3df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N,O);
    return output;
}
void cube_r_rnorm3df(float * x, float * y, float * z, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_rnorm3df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N,O);
}

__global__ void cube_rnorm4df_device(float * a, float * b, float * c, float * d, float* out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rnorm4df(a[idx],b[idx],c[idx],d[idx]);
}


float* cube_rnorm4df(float * x, float * y, float * z, float * q, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_rnorm4df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,q,output,M,N,O);
    return output;
}
void cube_r_rnorm4df(float * x, float * y, float * z, float * q, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_rnorm4df_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,q,output,M,N,O);
}


__global__ void cube_rnormf_device(int dim, const float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rnormf(dim,a);
}

float* cube_rnormf(int d, float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);                
    cube_rnormf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(d,x,output,M,N,O);
    return output;
}
void cube_r_rnormf(int d, float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_rnormf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(d,x,output,M,N,O);
}

__global__ void cube_scalblnf_device(float * a, long int m,float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = scalblnf(a[idx],m);
}


float* cube_scalblnf(float * x, long int m, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_scalblnf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,m,output,M,N,O);
    return output;
}
void cube_r_scalblnf(float * x, long int m, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_scalblnf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,m,output,M,N,O);
}

__global__ void cube_sinpif_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = sinpif(a[idx]);
}

float* cube_sinpif(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);                
    cube_sinpif_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_sinpif(float * x, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_sinpif_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O); 
}

__global__ void cube_tgammaf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = tgammaf(a[idx]);
}

float* cube_tgammaf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_tgammaf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_tgammaf(float * x, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_tgammaf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}


__global__ void cube_truncf_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = truncf(a[idx]);
}

float* cube_truncf(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);                cube_truncf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_truncf(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_truncf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}

__global__ void cube_y0f_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = y0f(a[idx]);
}

float* cube_y0f(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_y0f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_y0f(float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_y0f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}



__global__ void cube_y1f_device(float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = y1f(a[idx]);
}

float* cube_y1f(float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_y1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
    return output;
}
void cube_r_y1f(float * x, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_y1f_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N,O);
}


__global__ void cube_ynf_device(int m, float * a, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = ynf(m,a[idx]);
}

float* cube_ynf(int m, float * x, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_ynf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(m,x,output,M,N,O);
    return output;
}
void cube_r_ynf(int m, float * x, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_ynf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(m,x,output,M,N,O);
}

__global__ void cube_fdimf_const_device(float * a, float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fdimf(a[idx],b);
}

float* cube_fdimf_const(float * a, float  b, int M, int N, int O)
{
int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_fdimf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O);    
    return output;
}
void cube_r_fdimf_const(float * a, float  b, float * output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_fdimf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O);        
}


__global__ void cube_fdividef_const_device(float * a, float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = fdividef(a[idx],b);
}

float* cube_fdividef_const(float * a, float b, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_fdividef_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O);    
    return output;
}
void cube_r_fdividef_const(float * a, float b, float *output, int M, int N, int O)
{    
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_fdividef_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,O);     
}

__global__ void cube_hypotf_const_device(float * a,float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = hypotf(a[idx],b);
}

float* cube_hypotf_const(float * x, float y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_hypotf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}
void cube_r_hypotf_const(float * x, float y, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_hypotf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
}



__global__ void cube_remainderf_const_device(float * a, float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = remainderf(a[idx],b);
}


float* cube_remainderf_const(float * x, float y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_remainderf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}
void cube_r_remainderf_const(float * x, float y, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_remainderf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);        
}

__global__ void cube_rhypotf_const_device(float * a, float b, float * out, int M, int N, int O)
{
    int idxx = threadIdx.x + blockIdx.x * blockDim.x;     
    int idxy = threadIdx.y + blockIdx.y * blockDim.y;     
    int idxz = threadIdx.z + blockIdx.z * blockDim.z;     
    int idx = idxx*N + idxy*O + idxz;
    if(idxx < M && idxy < N && idxz < O)
        out[idx] = rhypotf(a[idx],b);
}

float* cube_rhypotf_const(float * x, float y, int M, int N, int O)
{
    int n = dim(M,N,O);
    float * output = find_memory(n);
    if(output == NULL) cudaMalloc((void**)&output,n*sizeof(float));    
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_rhypotf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
    return output;
}
void cube_r_rhypotf_const(float * x, float y, float *output, int M, int N, int O)
{
    int n = dim(M,N,O);
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_deep = (O + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols, grid_deep);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);            
    cube_rhypotf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N,O);    
}
