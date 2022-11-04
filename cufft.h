#include <cuda/std/complex>

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <complex>
#include <ccomplex>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cufft.h>

class cuFFT
{    
private:
    cufftHandle     plan;
    cufftComplex *  in;     
    cufftComplex *  host;   
    cufftComplex *  out;
    size_t          NX;
    size_t          NY;
    size_t          NZ;

public:

    cuFFT(int nx) {
        NX = nx;
        NY = 1;
        NZ = 1;
        cufftType type = CUFFT_C2C;
        cudaMalloc((void**)&in,sizeof(cufftComplex)*nx);     
        assert(in != NULL);
        cudaMalloc((void**)&out,sizeof(cufftComplex)*nx);     
        assert(out != NULL);        
        cufftPlan1d(&plan,NX,type,1);
        host = (cufftComplex*)calloc(NX,sizeof(cufftComplex));
        assert(host != NULL);
    }
    cuFFT(int nx, int ny) {
        NX = nx;
        NY = ny;
        NZ = 1;
        cufftType type = CUFFT_C2C;
        cudaMalloc((void**)&in,sizeof(cufftComplex)*nx*ny);     
        assert(in != NULL);
        cudaMalloc((void**)&out,sizeof(cufftComplex)*nx*ny);     
        assert(out != NULL);
        host = (cufftComplex*)calloc(NX*NY,sizeof(cufftComplex));
        assert(host != NULL);
        cufftPlan2d(&plan,NX,NY,type);        
    }
    cuFFT(int nx, int ny, int nz) {
        NX = nx;
        NY = ny;
        NZ = nz;
        cufftType type = CUFFT_C2C;
        cudaMalloc((void**)&in,sizeof(cufftComplex)*nx*ny*nz);     
        assert(in != NULL);        
        cudaMalloc((void**)&out,sizeof(cufftComplex)*nx*ny*nz);     
        assert(out != NULL);
        host = (cufftComplex*)calloc(NX*NY*NZ,sizeof(cufftComplex));
        assert(host != NULL);
        cufftPlan3d(&plan,NX,NY,NZ,type);
    }
    ~cuFFT() {
        if(in) cudaFree(in);
        if(out) cudaFree(out);                
    }

    void download_host() {
        cudaMemcpy(host,out,NX*NY*NZ*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
    }
    void upload_device() {
        cudaMemcpy(in,host,NX*NY*NZ*sizeof(cufftComplex),cudaMemcpyHostToDevice);
    }
    cufftComplex& operator[](size_t index) {
        return host[index];
    }
    cufftComplex __getitem(size_t index) { 
        return host[index];
    }
    void __setitem(size_t index, const cufftComplex & val) {
        host[index] = val;
    }    

    void forward() {
        upload_device();
        cufftExecC2C(plan,in,out,CUFFT_FORWARD);        
        synchronize();
        download_host();
    }
    void forward(std::vector<cufftComplex> & inputs) {
        memcpy(host,inputs.data(),inputs.size()*sizeof(cufftComplex));
        upload_device();
        cufftExecC2C(plan,in,out,CUFFT_FORWARD);        
        synchronize();
        download_host();
    }
    void forwardR2C(std::vector<float> & inputs) {
        for(size_t i = 0; i < inputs.size(); i++)  {
            ((float*)host)[i] = inputs[i];            
        }
        upload_device();
        cufftExecR2C(plan,(float*)in,out);        
        synchronize();
        download_host();
    }

    void inverse() {
        upload_device();
        cufftExecC2C(plan,in,out,CUFFT_INVERSE);        
        synchronize();
        download_host();
    }
    void inverseC2R(std::vector<float> & outputs) {
        upload_device();
        cufftExecC2R(plan,in,(float*)out);        
        outputs.resize(NX*NY*NZ);
        synchronize();
        download_host();
        for(size_t i = 0; i < NX*NY*NZ; i++)
            outputs[i] = ((float*)host)[i];
    }
    void get_complex_vector(std::vector<std::complex<float>> & out)
    {
        out.resize(NX*NY*NZ);
        for(size_t i = 0; i < NX*NY*NZ; i++) {        
            out[i].real(host[i].x);
            out[i].imag(host[i].y);
        }
    }
    void synchronize() {
        cudaDeviceSynchronize();
    }
};
