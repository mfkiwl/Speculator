//////////////////////////////////////////////////////////////////////////
// matrix
//////////////////////////////////////////////////////////////////////////
#include <cassert>
#include "cuda_runtime.h"
#include "math_constants.h"
#include "matrix_float.h"
#include "vector_float.h"


cudaStream_t get_cuda_stream();



/*
std::map<std::string,matrix_kernel1> matrix_map1;
std::map<std::string,matrix_kernel2> matrix_map2;
std::map<std::string,matrix_kernel3> matrix_map3;
std::map<std::string,matrix_kernel4> matrix_map4;

void register_matrix_kernel1(const char * name, matrix_kernel1 kernel) {
    // should assert if already exist?
    matrix_map1[name] = kernel;
}
void register_matrix_kernel2(const char * name, matrix_kernel2 kernel) {
    matrix_map2[name] = kernel;
}
void register_matrix_kernel3(const char * name, matrix_kernel3 kernel) {
    matrix_map3[name] = kernel;
}
void register_matrix_kernel4(const char * name, matrix_kernel4 kernel) {
    matrix_map4[name] = kernel;
}
float* execute_matrix_kernel1(const char * name, float * x, int n) {
    typename std::map<std::string,matrix_kernel1>::iterator i = matrix_map1.find(name);
    // assert or return NULL?
    if(i == matrix_map1.end()) return NULL;
    return (i->second)(x,n);
}
float* execute_matrix_kernel2(const char * name, float * x, float * y, int n) {
    typename std::map<std::string,matrix_kernel2>::iterator i = matrix_map2.find(name);
    // assert or return NULL?
    if(i == matrix_map2.end()) return NULL;
    return (i->second)(x,y,n);
}
float* execute_matrix_kernel3(const char * name, float * x, float * y, float * z, int n) {
    typename std::map<std::string,matrix_kernel3>::iterator i = matrix_map3.find(name);
    // assert or return NULL?
    if(i == matrix_map3.end()) return NULL;
    return (i->second)(x,y,z,n);
}
float* execute_matrix_kernel4(const char * name, float * x, float * y, float * z, float * w, int n) {
    typename std::map<std::string,matrix_kernel4>::iterator i = matrix_map4.find(name);
    // assert or return NULL?
    if(i == matrix_map4.end()) return NULL;
    return (i->second)(x,y,z,w,n);
}
*/


__global__ void gpu_2d_addf(float *a,float *b, float *c, int m, int n)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;    
    if( col < n && row < m) 
    {        
        c[row * n + col] = a[row * n + col] + b[row*n + col];
    }
} 

float* _2d_addf(float * a, float * b, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_2d_addf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N);
    return output;
}
void _2d_r_addf(float * a, float * b, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_2d_addf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N);
}

__global__ void gpu_2d_mulf(float *a,float *b, float *c, int m, int n)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;    
    if( col < n && row < m) 
    {        
        c[row * n + col] = a[row * n + col] * b[row*n + col];
    }
} 

float* _2d_mulf(float * a, float * b, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_2d_mulf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N);
    return output;
}
void _2d_r_mulf(float * a, float * b, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_2d_mulf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N);
}


__global__ void gpu_2d_subf(float *a,float *b, float *c, int m, int n)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;    
    if( col < n && row < m) 
    {        
        c[row * n + col] = a[row * n + col] - b[row*n + col];
    }
} 

float* _2d_subf(float * a, float * b, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_2d_subf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N);
    return output;
}
void _2d_r_subf(float * a, float * b, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_2d_subf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N);
}

__global__ void gpu_2d_divf(float *a,float *b, float *c, int m, int n)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;    
    if( col < n && row < m) 
    {        
        c[row * n + col] = a[row * n + col] / b[row*n + col];
    }
} 

float* _2d_divf(float * a, float * b, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_2d_divf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N);
    return output;
}
void _2d_r_divf(float * a, float * b, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_2d_divf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N);
}

__global__ void gpu_2d_modf(float *a,float *b, float *c, int m, int n)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;    
    if( col < n && row < m) 
    {        
        c[row * n + col] = fmodf(a[row * n + col],b[row*n + col]);
    }
} 

float* _2d_modf(float * a, float * b, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_2d_modf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N);
    return output;
}
void _2d_r_modf(float * a, float * b, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_2d_modf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N);
}



__global__ void gpu_matrix_addf(float *a,float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;    
    if( col < k && row < m) 
    {        
        c[row * n + col] = a[row * n + col] + b[row*n + col];
    }
} 
float* matrix_addf(float * a, float * b, int M, int N, int K)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_addf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,K);
    return output;
}
void matrix_r_addf(float * a, float * b, float *output, int M, int N, int K)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_addf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,K);
}

__global__ void gpu_matrix_subf(float *a,float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;    
    if( col < k && row < m) 
    {                
        c[row * n + col] = a[row * n + col] - b[row*n + col];
    }
} 

float* matrix_subf(float * a, float * b, int M, int N, int K)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_subf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,K);
    return output;
}

void matrix_r_subf(float * a, float * b, float *output, int M, int N, int K)
{  
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_subf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,K); 
}



__global__ void gpu_matrix_hadamardf(float *a,float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;    
    if( col < k && row < m) 
    {                
        c[row * n + col] = a[row * n + col] * b[row * k + col];
    }
} 
float* matrix_hadamardf(float * a, float * b, int M, int N, int K)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_hadamardf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,K);
    return output;
}


void matrix_r_hadamardf(float * a, float * b, float *output, int M, int N, int K)
{   
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_hadamardf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,K);
}


__global__ void gpu_matrix_multiplyf(float *a,float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;    
    
    if( col < k && row < m) 
    {     
        for(int i = 0; i < n; i++) 
        {            
            sum += a[row*n+i]*b[i*k+col];
        }        
        c[row*n + col] = sum;
    }

} 

float* matrix_multiplyf(float * a, float * b, int M, int N, int K)
{
    float * output = find_memory(M*K);
    if(output == NULL) cudaMalloc((void**)&output,M*K*sizeof(float));
    int BLOCK_SIZE=1024;
    if( N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows,grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_multiplyf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,K);
    return output;
}


void matrix_r_multiplyf(float * a, float * b, float * output, int M, int N, int K)
{
    int BLOCK_SIZE=1024;
    if( N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows,grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_multiplyf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(a,b,output,M,N,K);
}

__global__ void gpu_matrix_transposef(float* mat_in, float* mat_out, unsigned int rows, unsigned int cols) 
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        
        unsigned int pos        =  idx * rows + idy;
        unsigned int trans_pos  =  idy * cols + idx;
        mat_out[trans_pos] = mat_in[pos];    
    }
}


float* matrix_transposef(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_transposef<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}

void matrix_r_transposef(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_transposef<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_softmaxf(float* x, float* c, float sum, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = expf(x[row*n + col])/sum;        
    }
}

float* matrix_softmaxf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    float sum = vector_sumf(input,M*N);
    gpu_matrix_softmaxf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,sum,M,N);
    return output;
}

void matrix_r_softmaxf(float * input, float *output, int M, int N)
{   
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    float sum = vector_sumf(input,M*N);
    gpu_matrix_softmaxf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,sum,M,N);
}


__global__ void gpu_matrix_acosf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = acosf(x[row*n + col]);
    }
}

float* matrix_acosf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_acosf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_acosf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_acosf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_asinf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = asinf(x[row*n + col]);
    }
}

float* matrix_asinf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_asinf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}

void matrix_r_asinf(float * input, float * output,  int M, int N)
{    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_asinf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_atanf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = atanf(x[row*n + col]);
    }
}

float* matrix_atanf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_atanf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}

void matrix_r_atanf(float * input, float * output, int M, int N)
{    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_atanf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);    
}

__global__ void gpu_matrix_atan2f(float* x, float *y, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = atan2f(x[row*n + col],y[row*n+col]);
    }
}

float* matrix_atan2f(float * x, float * y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_atan2f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
    return output;
}

void matrix_r_atan2f(float * x, float * y, float * output, int M, int N)
{    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_atan2f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N); 
}


__global__ void gpu_matrix_cosf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = cosf(x[row*n + col]);
    }
}

float* matrix_cosf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_cosf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}

void matrix_r_cosf(float * input, float * output, int M, int N)
{   
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_cosf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N); 
}

__global__ void gpu_matrix_sinf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = sinf(x[row*n + col]);
    }
}

float* matrix_sinf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_sinf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}

void matrix_r_sinf(float * input, float * output, int M, int N)
{   
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_sinf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N); 
}

__global__ void gpu_matrix_tanf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = tanf(x[row*n + col]);
    }
}

float* matrix_tanf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_tanf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}

void matrix_r_tanf(float * input, float * output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_tanf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_coshf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = coshf(x[row*n + col]);
    }
}

float* matrix_coshf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_coshf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}

void matrix_r_coshf(float * input, float * output, int M, int N)
{    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_coshf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N); 
}

__global__ void gpu_matrix_sinhf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = sinhf(x[row*n + col]);
    }
}

float* matrix_sinhf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_sinhf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}

void matrix_r_sinhf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_sinhf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_tanhf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = tanhf(x[row*n + col]);
    }
}

float* matrix_tanhf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_tanhf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}


void matrix_r_tanhf(float * input, float * output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_tanhf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_acoshf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = acoshf(x[row*n + col]);
    }
}

float* matrix_acoshf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_acoshf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}

void matrix_r_acoshf(float * input, float * output, int M, int N)
{    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_acoshf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N); 
}

__global__ void gpu_matrix_asinhf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = asinhf(x[row*n + col]);
    }
}

float* matrix_asinhf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_asinhf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}

void matrix_r_asinhf(float * input, float * output, int M, int N)
{   
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_asinhf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N); 
}

__global__ void gpu_matrix_atanhf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = atanhf(x[row*n + col]);
    }
}

float* matrix_atanhf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_atanhf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}

void matrix_r_atanhf(float * input, float *output, int M, int N)
{  
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_atanhf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N); 
}


__global__ void gpu_matrix_ceilf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = ceilf(x[row*n + col]);
    }
}

float* matrix_ceilf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_ceilf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_ceilf(float * input, float *output, int M, int N)
{  
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_ceilf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N); 
}



__global__ void gpu_matrix_exp10f(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = exp10f(x[row*n + col]);
    }
}

float* matrix_exp10f(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_exp10f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_exp10f(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_exp10f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    
}

__global__ void gpu_matrix_exp2f(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = exp2f(x[row*n + col]);
    }
}

float* matrix_exp2f(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_exp2f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_exp2f(float * input, float *output, int M, int N)
{  
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_exp2f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N); 
}

__global__ void gpu_matrix_expf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col;
    if (col < n && row < m)     {        
        c[idx] = expf(x[idx]);
    }    
}

float* matrix_expf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);        
    gpu_matrix_expf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);    
    return output;
}

void matrix_r_expf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_expf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_expm1f(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = expm1f(x[row*n + col]);
    }
}

float* matrix_expm1f(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_expm1f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_expm1f(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_expm1f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_fabsf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = fabsf(x[row*n + col]);
    }
}

float* matrix_fabsf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fabsf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_fabsf(float * input, float *output, int M, int N)
{   
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fabsf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N); 
}


__global__ void gpu_matrix_floorf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = floorf(x[row*n + col]);
    }
}

float* matrix_floorf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));   
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_floorf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_floorf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_floorf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_fmaxf(float* x, float * y, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = fmaxf(x[row*n + col],y[row*n + col]);
    }
}

float* matrix_fmaxf(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fmaxf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
    return output;
}
void matrix_r_fmaxf(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fmaxf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
}

__global__ void gpu_matrix_fminf(float* x, float * y, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = fminf(x[row*n + col],y[row*n + col]);
    }
}

float* matrix_fminf(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fminf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
    return output;
}
void matrix_r_fminf(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fminf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void gpu_matrix_fmodf(float* x, float * y, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = fmodf(x[row*n + col],y[row*n + col]);
    }
}

float* matrix_fmodf(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fmodf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
    return output;
}
void matrix_r_fmodf(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fmodf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
}



__global__ void gpu_matrix_log10f(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = log10f(x[row*n + col]);
    }
}

float* matrix_log10f(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_log10f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_log10f(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_log10f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_log1pf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = log1pf(x[row*n + col]);
    }
}

float* matrix_log1pf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_log1pf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_log1pf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_log1pf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_log2f(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = log2f(x[row*n + col]);
    }
}

float* matrix_log2f(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_log2f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_log2f(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_log2f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_logbf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = logbf(x[row*n + col]);
    }
}

float* matrix_logbf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_logbf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_logbf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_logbf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}



__global__ void gpu_matrix_powf(float* x, float * y, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = powf(x[row*n + col],y[row*n + col]);
    }
}

float* matrix_powf(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_powf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
    return output;
}
void matrix_r_powf(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_powf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
}


__global__ void gpu_matrix_rsqrtf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = rsqrtf(x[row*n + col]);
    }
}

float* matrix_rsqrtf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_rsqrtf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_rsqrtf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_rsqrtf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_sqrtf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = sqrtf(x[row*n + col]);
    }
}

float* matrix_sqrtf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_sqrtf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_sqrtf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_sqrtf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}






__global__ void matrix_sigmoid_device(float * x, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        float r = 1.0f / (1.0f + expf(-x[idx]));
        out[idx] = r;
    }
}

float* matrix_sigmoidf(float * x, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));             
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);       
    matrix_sigmoid_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N);        
    return output;
}
void matrix_r_sigmoidf(float * x, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);       
    matrix_sigmoid_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N);        
}

__global__ void matrix_sigmoid_grad_device(float * x, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        float r = x[idx] * (1.0f - x[idx]);
        out[idx] = r;
    }    
}


float* matrix_sigmoid_gradf(float * x, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_sigmoid_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N);        
    return output;
}
void matrix_r_sigmoid_gradf(float * x, float *output,int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_sigmoid_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N);        
}


__global__ void matrix_tanh_grad_device(float * x, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = 1.0 - (x[idx] *x[idx]);
    }    
}

float* matrix_tanh_gradf(float * x, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_tanh_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N);        
    return output;
}
void matrix_r_tanh_gradf(float * x, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_tanh_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N);        
}

__global__ void matrix_relu_device(float * x, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        if(x[idx] < 0) out[idx] = 0.0f;
        else out[idx] = x[idx];        
    }
}    

float* matrix_reluf(float * x, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_relu_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N);        
    return output;
}
void matrix_r_reluf(float * x, float *output, int M, int N)
{    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_relu_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N);         
}

__global__ void matrix_relu_grad_device(float * x, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        if(x[idx] > 0) out[idx] = 1.0;
        else out[idx] = 0.0f;
    }
}

float* matrix_relu_gradf(float * x, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_relu_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N);        
    return output;
}
void matrix_r_relu_gradf(float * x, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_relu_grad_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N);        
}

__global__ void matrix_add_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = x[idx] + y;
    }
}

float* matrix_addf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_add_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_addf_const(float * x, float y, float *output, int M, int N)
{    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_add_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);     
}

__global__ void matrix_sub_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = x[idx] - y;
    }
}

float* matrix_subf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_sub_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_subf_const(float * x, float y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_sub_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_mul_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = x[idx] * y;
    }
    __syncthreads();
}

float* matrix_mulf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_mul_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_mulf_const(float * x, float y, float *output,int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_mul_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_div_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m && y != 0.0f) 
    {        
        out[idx] = x[idx] / y;
    }
}

float* matrix_divf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    assert(y != 0.0f);    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_div_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_divf_const(float * x, float y, float *output, int M, int N)
{
    assert(y != 0.0f);
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_div_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_mod_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = fmod(x[idx],y);
    }
}

float* matrix_modf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_mod_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_modf_const(float * x, float y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_mod_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_atan2f_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = atan2f(x[idx],y);
    }
}

float* matrix_atan2f_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_atan2f_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}

void matrix_r_atan2f_const(float * x, float y, float * output, int M, int N)
{    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_atan2f_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);     
}


__global__ void matrix_fmaxf_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = fmaxf(x[idx],y);
    }
}

float* matrix_fmaxf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fmaxf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_fmaxf_const(float * x, float y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fmaxf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_fminf_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = fminf(x[idx],y);
    }
}

float* matrix_fminf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fminf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_fminf_const(float * x, float y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fminf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_fmodf_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = fmodf(x[idx],y);
    }
}

float* matrix_fmodf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fmodf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_fmodf_const(float * x, float y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fmodf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_powf_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = powf(x[idx],y);
    }
}

float* matrix_powf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_powf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_powf_const(float * x, float y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_powf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}



__global__ void matrix_add_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = x[idx] +y[0];
    }
}

float* matrix_addf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_add_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_addf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_add_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_subf_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = x[idx] - y[0];
    }
}

float* matrix_subf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_add_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_subf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_add_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_mul_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = x[idx] * y[0];
    }
}

float* matrix_mulf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_mul_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_mulf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_mul_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_div_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        if(y[0] != 0.0)
            out[idx] = x[idx] / y[0];
        // if there is some reason not to do this, it can be changed.
        else 
            out[idx] = CUDART_NAN_F;
    }
}

float* matrix_divf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_div_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}

void matrix_r_divf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_div_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_mod_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {            
        out[idx] = fmodf(x[idx],y[0]);        
    }
}

float* matrix_modf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_mod_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_modf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_mod_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_fmax_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {            
        out[idx] = fmaxf(x[idx],y[0]);        
    }
}

float* matrix_fmaxf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fmax_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_fmaxf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fmax_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_fmin_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {            
        out[idx] = fminf(x[idx],y[0]);        
    }
}

float* matrix_fminf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fmin_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_fminf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fmin_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_pow_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {            
        out[idx] = powf(x[idx],y[0]);        
    }
}

float* matrix_powf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_pow_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_powf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_pow_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_hypot_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {            
        out[idx] = hypotf(x[idx],y[0]);        
    }
}

float* matrix_hypotf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_hypot_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_hypotf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_hypot_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_rhypot_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {            
        out[idx] = rhypotf(x[idx],y[0]);        
    }
}

float* matrix_rhypotf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_rhypot_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_rhypotf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_rhypot_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_fdividef_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {            
        out[idx] = fdividef(x[idx],y[0]);        
    }
}

float* matrix_fdividef_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fdividef_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_fdividef_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fdividef_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_fmodf_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {            
        out[idx] = fmodf(x[idx],y[0]);        
    }
}

float* matrix_fmodf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fmodf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_fmodf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fmodf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_remainderf_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {            
        out[idx] = remainderf(x[idx],y[0]);        
    }
}

float* matrix_remainderf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_remainderf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_remainderf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_remainderf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_fdimf_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {            
        out[idx] = fdimf(x[idx],y[0]);        
    }
}

float* matrix_fdimf_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fdimf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_fdimf_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fdimf_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_atan2f_scalar_device(float * x, float *y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {            
        out[idx] = atan2f(x[idx],y[0]);        
    }
}

float* matrix_atan2f_scalar(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_atan2f_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_atan2f_scalar(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_atan2f_scalar_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

////////////////////////////////////////////////////////////////////////////////
// matrix

__global__ void gpu_matrix_cbrtf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = cbrtf(x[row*n + col]);
    }
}

float* matrix_cbrtf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_cbrtf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_cbrtf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_cbrtf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_cospif(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = cospif(x[row*n + col]);
    }
}

float* matrix_cospif(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_cospif<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_cospif(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_cospif<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_cyl_bessel_i0f(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = cyl_bessel_i0f(x[row*n + col]);
    }
}

float* matrix_cyl_bessel_i0f(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_cyl_bessel_i0f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_cyl_bessel_i0f(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_cyl_bessel_i0f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_cyl_bessel_i1f(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = cyl_bessel_i1f(x[row*n + col]);
    }
}

float* matrix_cyl_bessel_i1f(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_cyl_bessel_i1f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_cyl_bessel_i1f(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_cyl_bessel_i1f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_erfcf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = erfcf(x[row*n + col]);
    }
}

float* matrix_erfcf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_erfcf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_erfcf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_erfcf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_erfcinvf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = erfcinvf(x[row*n + col]);
    }
}

float* matrix_erfcinvf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_erfcinvf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_erfcinvf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_erfcinvf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_erfcxf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = erfcxf(x[row*n + col]);
    }
}

float* matrix_erfcxf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_erfcxf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_erfcxf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_erfcxf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_erff(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = erff(x[row*n + col]);
    }
}

float* matrix_erff(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_erff<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_erff(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_erff<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_erfinvf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = erfinvf(x[row*n + col]);
    }
}

float* matrix_erfinvf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_erfinvf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_erfinvf(float * input, float * output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_erfinvf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_fdimf(float* x, float * y, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = fdimf(x[row*n + col],y[row*n + col]);
    }
}

float* matrix_fdimf(float * x, float * y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fdimf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
    return output;
}
void matrix_r_fdimf(float * x, float * y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fdimf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
}

__global__ void gpu_matrix_fdividef(float* x, float * y, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = fdividef(x[row*n + col],y[row*n + col]);
    }
}

float* matrix_fdividef(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fdividef<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
    return output;
}
void matrix_r_fdividef(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fdividef<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
}

__global__ void gpu_matrix_fmaf(float* x, float * y, float *z, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = fmaf(x[row*n + col],y[row*n + col],z[row*n + col]);
    }
}

float* matrix_fmaf(float * x, float *y, float *z, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fmaf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N);
    return output;
}
void matrix_r_fmaf(float * x, float *y, float *z, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_fmaf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N);
}

__global__ void gpu_matrix_hypotf(float* x, float * y, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = hypotf(x[row*n + col],y[row*n + col]);
    }
}

float* matrix_hypotf(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_hypotf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
    return output;
}
void matrix_r_hypotf(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_hypotf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
}

__global__ void gpu_matrix_ilogbf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = ilogbf(x[row*n + col]);
    }
}

float* matrix_ilogbf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_ilogbf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_ilogbf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_ilogbf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_j0f(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = j0f(x[row*n + col]);
    }
}

float* matrix_j0f(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_j0f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_j0f(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_j0f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_j1f(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = j1f(x[row*n + col]);
    }
}

float* matrix_j1f(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_j1f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_j1f(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_j1f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_jnf(float* x, int N, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = jnf(x[row*n + col],N);
    }
}

float* matrix_jnf(float * input, int n, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_jnf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,n,output,M,N);
    return output;
}
void matrix_r_jnf(float * input, float *output, int n, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_jnf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,n,output,M,N);
}


__global__ void gpu_matrix_ldexpf(float* x, int exp, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = ldexpf(x[row*n + col],exp);
    }
}

float* matrix_ldexpf(float * input, int exp, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_ldexpf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,exp,output,M,N);
    return output;
}
void matrix_r_ldexpf(float * input, float *output, int exp, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_ldexpf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,exp,output,M,N);
}

__global__ void gpu_matrix_lgammaf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = lgammaf(x[row*n + col]);
    }
}

float* matrix_lgammaf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_lgammaf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_lgammaf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_lgammaf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_copysign(float* x, float *y, float* c, int m, int n) 
{    
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col;
    if (col < n && row < m) 
    {        
        c[idx] = copysignf(x[idx],y[idx]);
    }
}

float* matrix_copysignf(float * X, float *Y, int M , int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_copysign<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(X,Y,output,M,N);
    return output;
}
void matrix_r_copysignf(float * X, float *Y, float *output, int M, int N)
{    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_copysign<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(X,Y,output,M,N);
}


__global__ void gpu_matrix_nearbyintf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = nearbyintf(x[row*n + col]);
    }
}

float* matrix_nearbyintf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_nearbyintf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_nearbyintf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_nearbyintf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_norm3df(float* x, float *y, float *z, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = norm3df(x[row*n + col],y[row*n + col],z[row*n + col]);
    }
}

float* matrix_norm3df(float * x, float *y, float *z, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_norm3df<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N);
    return output;
}
void matrix_r_norm3df(float * x, float *y, float *z, float *output, int M, int N)
{   
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_norm3df<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N); 
}


__global__ void gpu_matrix_norm4df(float* x, float *y, float *z, float * w, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = norm4df(x[row*n + col],y[row*n + col],z[row*n + col],w[row*n + col]);
    }
}

float* matrix_norm4df(float * x, float *y, float *z, float * w, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_norm4df<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,w,output,M,N);
    return output;
}
void matrix_r_norm4df(float * x, float *y, float *z, float * w, float *output,int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_norm4df<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,w,output,M,N);
}

__global__ void gpu_matrix_normcdff(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = normcdff(x[row*n + col]);
    }
}

float* matrix_normcdff(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_normcdff<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_normcdff(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_normcdff<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_normcdfinvf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = normcdfinvf(x[row*n + col]);
    }
}

float* matrix_normcdfinvf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_normcdfinvf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_normcdfinvf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_normcdfinvf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_normf(int dim, const float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = normf(dim, x);
    }
}

float* matrix_normf(int dim, float *input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_normf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(dim,input,output,M,N);
    return output;
}
void matrix_r_normf(int dim, float *input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_normf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(dim,input,output,M,N);
}

__global__ void gpu_matrix_remainderf(float* x, float * y, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = remainderf(x[row*n + col],y[row*n + col]);
    }
}

float* matrix_remainderf(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_remainderf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
    return output;
}
void matrix_r_remainderf(float * x, float *y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_remainderf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
}




__global__ void gpu_matrix_rcbrtf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = rcbrtf(x[row*n + col]);
    }
}

float* matrix_rcbrtf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_rcbrtf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_rcbrtf(float * input, float * output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_rcbrtf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}



__global__ void gpu_matrix_rhypotf(float* x, float *y, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = rhypotf(x[row*n + col],y[row*n + col]);
    }
}

float* matrix_rhypotf(float * x, float *y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_rhypotf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
    return output;
}
void matrix_r_rhypotf(float * x, float *y, float * output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_rhypotf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);
}

__global__ void gpu_matrix_rnorm3df(float* x, float *y, float *z, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = rnorm3df(x[row*n + col],y[row*n + col],z[row*n + col]);
    }
}

float* matrix_rnorm3df(float * x, float *y, float *z, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_rnorm3df<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N);
    return output;
}

void matrix_r_rnorm3df(float * x, float *y, float *z, float * output, int M, int N)
{    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_rnorm3df<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,output,M,N); 
}


__global__ void gpu_matrix_rnorm4df(float* x, float *y, float *z, float *w, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = rnorm4df(x[row*n + col],y[row*n + col],z[row*n + col],w[row*n + col]);
    }
}

float* matrix_rnorm4df(float * x, float *y, float *z, float *w, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_rnorm4df<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,w,output,M,N);
    return output;
}
void matrix_r_rnorm4df(float * x, float *y, float *z, float *w, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_rnorm4df<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,z,w,output,M,N);
}



__global__ void gpu_matrix_rnormf(int dim, const float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = rnormf(dim, x);
    }
}

float* matrix_rnormf(int dim, float *input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_rnormf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(dim,input,output,M,N);
    return output;
}
void matrix_r_rnormf(int dim, float *input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_rnormf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(dim,input,output,M,N);    
}


__global__ void gpu_matrix_scalblnf(float* x, long int N, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = scalblnf(x[row*n + col],N);
    }
}       
float* matrix_scalblnf(float * input, long int n, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_scalblnf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,n,output,M,N);
    return output;
}
void matrix_r_scalblnf(float * input, long int n, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_scalblnf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,n,output,M,N);
}

__global__ void gpu_matrix_sinpif(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = sinpif(x[row*n + col]);
    }
}

float* matrix_sinpif(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_sinpif<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_sinpif(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_sinpif<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_tgammaf(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = tgammaf(x[row*n + col]);
    }
}

float* matrix_tgammaf(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_tgammaf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_tgammaf(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_tgammaf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_y0f(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = y0f(x[row*n + col]);
    }
}

float* matrix_y0f(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_y0f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_y0f(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_y0f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}


__global__ void gpu_matrix_y1f(float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = y1f(x[row*n + col]);
    }
}

float* matrix_y1f(float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_y1f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
    return output;
}
void matrix_r_y1f(float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_y1f<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(input,output,M,N);
}

__global__ void gpu_matrix_ynf(int N,float* x, float* c, int m, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n && row < m) 
    {        
        c[row*n + col] = ynf(N,x[row*n + col]);
    }
}

float* matrix_ynf(int n, float * input, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_ynf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(n,input,output,M,N);
    return output;
}
void matrix_r_ynf(int n, float * input, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    gpu_matrix_ynf<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(n,input,output,M,N);
}

__global__ void matrix_fdimf_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = fdimf(x[idx],y);
    }
}

float* matrix_fdimf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fdimf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_fdimf_const(float * x, float y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fdimf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_fdividef_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = fdividef(x[idx],y);
    }
}

float* matrix_fdividef_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fdividef_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_fdividef_const(float * x, float y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_fdividef_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_hypotf_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = hypotf(x[idx],y);
    }
}

float* matrix_hypotf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_hypotf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_hypotf_const(float * x, float y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_hypotf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}


__global__ void matrix_remainderf_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = remainderf(x[idx],y);
    }
}

float* matrix_remainderf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_remainderf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_remainderf_const(float * x, float y, float * output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_remainderf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_rhypotf_const_device(float * x, float y, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = fmaxf(x[idx],y);
    }
}

float* matrix_rhypotf_const(float * x, float y, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));
    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_rhypotf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
    return output;
}
void matrix_r_rhypotf_const(float * x, float y, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_rhypotf_const_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,y,output,M,N);    
}

__global__ void matrix_truncf_device(float * a, float * out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row*n + col; 
    if (col < n && row < m) 
    {        
        out[idx] = truncf(a[idx]);
    }
}

float* matrix_truncf(float * x, int M, int N)
{
    float * output = find_memory(M*N);
    if(output == NULL) cudaMalloc((void**)&output,M*N*sizeof(float));    
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_truncf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N);
    return output;
}
void matrix_r_truncf(float * x, float *output, int M, int N)
{
    int BLOCK_SIZE=1024;
    if(N < 1024) BLOCK_SIZE=N;
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);    
    matrix_truncf_device<<<dimGrid,dimBlock,0,get_cuda_stream()>>>(x,output,M,N);
}
