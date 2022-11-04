// add vector cumaths
// jitify/nvrtc
// is working

#ifndef VIPERFISH_H
#define VIPERFISH_H

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <memory>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"
//#include "cutensor.h"

#include "vector_float.h"
#include "matrix_float.h"
#include "cube_float.h"
#include "field_float.h"
#include "array_float.h"
#include "csvparser.h"

#define IDX2(i,j,n) ((j*n)+i)
#define IDX3(i,j,k,n,o) ((k*o)+(j*n)+i)
#define IDX4(i,j,k,w,n,o,p) ((w*p)+(k*o)+(j*n)+i)


#define ASSERTOK(status) (assert(status == CUBLAS_STATUS_SUCCESS))


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuda/cublas
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct CudaStream
{
    cudaStream_t stream;

    CudaStream() {
        cudaStreamCreate(&stream);
    }
    ~CudaStream() {
        cudaStreamDestroy(stream);
    }
};



struct CublasPointerMode 
{
    cublasPointerMode_t pointer_mode;
};

struct CublasAtomicsMode 
{
    cublasAtomicsMode_t atomics_mode;
};

struct CublasMathMode 
{
    cublasMath_t math_mode;
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Cublas 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Cublas
{
    cublasStatus_t   status;
    cublasHandle_t   handle;

    Cublas()
    {        
        status = cublasCreate(&handle);
        ASSERTOK(status);
    }
    ~Cublas() 
    {
        if(handle) cublasDestroy(handle);
    }    

    int GetVersion() 
    {
        int v = -1;
        status = cublasGetVersion(handle,&v);
        ASSERTOK(status);
        return v;
    }

    const char* GetStatusName()
    {
        const char * r  = cublasGetStatusName(status);
        return r;
    }
    
    void SetWorkspace(void * workspace, size_t workspace_size)
    {
        status = cublasSetWorkspace(handle,workspace,workspace_size);
        ASSERTOK(status);
    }

    void SetStream(const CudaStream& stream)
    {
        status = cublasSetStream(handle,stream.stream);
        ASSERTOK(status);
    }

    void GetStream(CudaStream & stream)
    {
        status = cublasGetStream(handle,&stream.stream);
        ASSERTOK(status);
    }

    void SetPointerMode(CublasPointerMode &p)
    {
        status = cublasSetPointerMode(handle,p.pointer_mode);
        ASSERTOK(status);
    }
    void GetPointerMode(CublasPointerMode & p)
    {
        status = cublasGetPointerMode(handle,&p.pointer_mode);
        ASSERTOK(status);
    }
    void SetAtomicsMode(CublasAtomicsMode & a)
    {
        status = cublasSetAtomicsMode(handle,a.atomics_mode);
        ASSERTOK(status);
    }
    void GetAtomicsMode(CublasAtomicsMode & a)
    {
        status = cublasGetAtomicsMode(handle,&a.atomics_mode);
        ASSERTOK(status);
    }
    void SetMathMode(CublasMathMode & m)
    {
        status = cublasSetMathMode(handle,m.math_mode);
        ASSERTOK(status);
    }
    void GetMathMode(CublasMathMode & m)
    {
        status = cublasGetMathMode(handle,&m.math_mode);
        ASSERTOK(status);
    }
    void SetSmCountTarget(int countTarget)
    {
        status = cublasSetSmCountTarget(handle,countTarget);
        ASSERTOK(status);
    }
    int GetSmCountTarget()
    {
        int sm = -1;
        status = cublasGetSmCountTarget(handle,&sm);
        ASSERTOK(status);
        return sm;
    }

    void LoggerConfigure(int logIsOn,int logToStdOut, int logToStdErr, const char * filename)
    {
        status = cublasLoggerConfigure(logIsOn, logToStdOut, logToStdErr, filename);
    }
};

extern Cublas *cublas;
cudaStream_t get_cuda_stream();
void set_stream(int streamid);
int get_stream();
cudaStream_t random_stream();
void    clear_cache();


typedef int array_index;

struct GPUIndex {
    array_index i;
    array_index j;
    array_index k;
    array_index w;

    GPUIndex(array_index m, array_index n=0, array_index o=0, array_index p=0) {
        i = m;
        j = n;
        k = o;
        w = p;
    }
};

enum GPUObjectType_t
{
    GPU_CONST,
    GPU_SCALAR,
    GPU_ARRAY,
    GPU_VECTOR,
    GPU_MATRIX,
    GPU_CUBE,
    GPU_FIELD,
};

class GPUObject 
{
private:
    
public:

    GPUObject() = default;
    virtual ~GPUObject() = default;

    virtual float * DP() const = 0;
    virtual size_t  size() const = 0;

    virtual GPUObjectType_t type() const = 0;

protected:

};

class GPUCube : public GPUObject 
{
    GPUCube() = default;
    ~GPUCube() = default;

    float *DP() { return nullptr; }
    size_t size() const { return 0; }
    GPUObjectType_t type() const { return GPU_CUBE; }

    size_t Dim1() const { return 0; }
    size_t Dim2() const { return 0; }
    size_t Dim3() const { return 0; }
    size_t Dim4() const { return 0; }

};
class GPUField : public GPUObject 
{
    GPUField() = default;
    ~GPUField() = default;

    float *DP() { return nullptr; }
    size_t size() const { return 0; }
    GPUObjectType_t type() const { return GPU_FIELD; }

    size_t Dim1() const { return 0; }
    size_t Dim2() const { return 0; }
    size_t Dim3() const { return 0; }
    size_t Dim4() const { return 0; }
};

struct CuRand 
{
    // curand 
    curandGenerator_t gen;

    CuRand(unsigned long long seed=0) {
        if(seed == 0)
            curandSetPseudoRandomGeneratorSeed(gen,time(NULL));
        else 
            curandSetPseudoRandomGeneratorSeed(gen,seed);
        curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    }
    ~CuRand() {
        curandDestroyGenerator(gen);
    }
    void curand_uniform(GPUObject& o) {                
        curandGenerateUniform(gen,o.DP(),o.size()*sizeof(float));                
    }
    void curand_normal(GPUObject &o, float mean, float stddev) {                
        curandGenerateNormal(gen,o.DP(),o.size()*sizeof(float), mean, stddev);                
    }
    void curand_lognormal(GPUObject &o, float mean, float stddev) {                
        curandGenerateLogNormal(gen,o.DP(),o.size()*sizeof(float), mean, stddev);                
    }    
};



struct Scalar {
    float value;
    Scalar(float x) : value(x) {}

    /*
    GPUArray operator + (const GPUArray & c);
    GPUArray operator - (const GPUArray & c);
    GPUArray operator * (const GPUArray & c);
    GPUArray operator / (const GPUArray & c);

    GPUVector operator + (const GPUVector & c);
    GPUVector operator - (const GPUVector & c);
    GPUVector operator * (const GPUVector & c);
    GPUVector operator / (const GPUVector & c);

    GPUMatrix operator + (const GPUMatrix & c);
    GPUMatrix operator - (const GPUMatrix & c);
    GPUMatrix operator * (const GPUMatrix & c);
    GPUMatrix operator / (const GPUMatix & c);

    GPUCube operator + (const GPUCube & c);
    GPUCube operator - (const GPUCube & c);
    GPUCube operator * (const GPUCube & c);
    GPUCube operator / (const GPUCube & c);

    GPUField operator + (const GPUField & c);
    GPUField operator - (const GPUField & c);
    GPUField operator * (const GPUField & c);
    GPUField operator / (const GPUField & c);    
    */

};


// this is going to be removed
class GPUConst : public GPUObject
{
public:

    float*                       devPtr;

    GPUConst(float value) {
        float * temp;
        cudaError_t err = cudaMalloc((void**)&temp,sizeof(float));
        if(err != cudaSuccess) {
            std::cout << cudaGetErrorString(err) << std::endl;
        }
        assert(err == cudaSuccess);                  
        devPtr = temp;        
    }   

    float* DP() const { return devPtr; }
    size_t size() const { return 1; }
    GPUObjectType_t type() const { return GPU_CONST; }

    /*
    GPUScalar operator + (const float c);
    GPUScalar operator - (const float c);
    GPUScalar operator * (const float c);
    GPUScalar operator / (const float c);

    GPUScalar operator + (const GPUConst & c);
    GPUScalar operator - (const GPUConst & c);
    GPUScalar operator * (const GPUConst & c);
    GPUScalar operator / (const GPUConst & c);

    GPUArray operator + (const GPUArray & c);
    GPUArray operator - (const GPUArray & c);
    GPUArray operator * (const GPUArray & c);
    GPUArray operator / (const GPUArray & c);

    GPUVector operator + (const GPUVector & c);
    GPUVector operator - (const GPUVector & c);
    GPUVector operator * (const GPUVector & c);
    GPUVector operator / (const GPUVector & c);

    GPUMatrix operator + (const GPUMatrix & c);
    GPUMatrix operator - (const GPUMatrix & c);
    GPUMatrix operator * (const GPUMatrix & c);
    GPUMatrix operator / (const GPUMatix & c);

    GPUCube operator + (const GPUCube & c);
    GPUCube operator - (const GPUCube & c);
    GPUCube operator * (const GPUCube & c);
    GPUCube operator / (const GPUCube & c);

    GPUField operator + (const GPUField & c);
    GPUField operator - (const GPUField & c);
    GPUField operator * (const GPUField & c);
    GPUField operator / (const GPUField & c);

    GPUScalar operator + (const GPUScalar & c);
    GPUScalar operator - (const GPUScalar & c);
    GPUScalar operator * (const GPUScalar & c);
    GPUScalar operator / (const GPUScalar & c);
    */

};

class GPUScalar : public GPUObject
{    
public:
    float* devPtr;

    GPUScalar(float value) {
        float * temp;
        cudaError_t err = cudaMalloc((void**)&temp,sizeof(float));
        if(err != cudaSuccess) {
            std::cout << cudaGetErrorString(err) << std::endl;
        }
        assert(err == cudaSuccess);                  
        devPtr = temp;
    }
    GPUScalar(float * p_dev) {
        devPtr = p_dev;        
    }

    float* DP() const { return devPtr; }
    size_t size() const { return 1; }
    GPUObjectType_t type() const { return GPU_SCALAR; }

    GPUScalar& operator = (const GPUScalar &s) {        
        devPtr = s.devPtr;        
        return *this;
    }
    void set_value(const float v) {
        cudaMemcpyAsync(devPtr,&v,sizeof(float),cudaMemcpyHostToDevice, ::get_cuda_stream());
    }
    float get_value() {
        float r;
        cudaMemcpyAsync(&r,devPtr,sizeof(float),cudaMemcpyDeviceToHost, ::get_cuda_stream());
        return r;
    }    

    GPUScalar operator + (const GPUScalar & s) {
        float * p = vector_addf(devPtr,s.devPtr,1);
        return GPUScalar(p);
    }
    GPUScalar operator - (const GPUScalar & s) {
        float * p = vector_subf(devPtr,s.devPtr,1);
        return GPUScalar(p);
    }
    GPUScalar operator * (const GPUScalar & s) {
        float * p = vector_mulf(devPtr,s.devPtr,1);
        return GPUScalar(p);
    }
    GPUScalar operator / (const GPUScalar & s) {
        float * p = vector_divf(devPtr,s.devPtr,1);
        return GPUScalar(p);
    }
    GPUScalar operator % (const GPUScalar & s) {
        float * p = vector_modf(devPtr,s.devPtr,1);
        return GPUScalar(p);
    }

    /*
    GPUScalar operator + (const float c);
    GPUScalar operator - (const float c);
    GPUScalar operator * (const float c);
    GPUScalar operator / (const float c);

    GPUScalar operator + (const GPUConst & c);
    GPUScalar operator - (const GPUConst & c);
    GPUScalar operator * (const GPUConst & c);
    GPUScalar operator / (const GPUConst & c);

    GPUArray operator + (const GPUArray & c);
    GPUArray operator - (const GPUArray & c);
    GPUArray operator * (const GPUArray & c);
    GPUArray operator / (const GPUArray & c);

    GPUVector operator + (const GPUVector & c);
    GPUVector operator - (const GPUVector & c);
    GPUVector operator * (const GPUVector & c);
    GPUVector operator / (const GPUVector & c);

    GPUMatrix operator + (const GPUMatrix & c);
    GPUMatrix operator - (const GPUMatrix & c);
    GPUMatrix operator * (const GPUMatrix & c);
    GPUMatrix operator / (const GPUMatix & c);

    GPUCube operator + (const GPUCube & c);
    GPUCube operator - (const GPUCube & c);
    GPUCube operator * (const GPUCube & c);
    GPUCube operator / (const GPUCube & c);

    GPUField operator + (const GPUField & c);
    GPUField operator - (const GPUField & c);
    GPUField operator * (const GPUField & c);
    GPUField operator / (const GPUField & c);

    GPUScalar operator + (const GPUScalar & c);
    GPUScalar operator - (const GPUScalar & c);
    GPUScalar operator * (const GPUScalar & c);
    GPUScalar operator / (const GPUScalar & c);
    */
    
};


struct GPUVector 
{
    float * devPtr;
    float * host;
    std::shared_ptr<float> host_ptr;
    size_t  N;

    GPUVector() {
        N=0;                
        devPtr = nullptr;
        host   = nullptr;
    }
    GPUVector(size_t n)    {
        assert(n > 0);        
        N = n;        
        init();
    }
    GPUVector(const std::vector<float> & data)    {        
        N = data.size();                
        init();
        cudaMemcpyAsync(devPtr,data.data(),size()*sizeof(float),cudaMemcpyHostToDevice, ::get_cuda_stream());
    }
    // p is on device
    GPUVector(float * devptr, size_t n, size_t x=0,size_t y=0, size_t z=0)   {
        assert(n > 0);
        N = n;        
        host = (float*)calloc(size(),sizeof(float));                
        devPtr = devptr;        
    }
    GPUVector(const GPUVector & a) {        
        N = a.N;
        init();
        cudaMemcpyAsync(devPtr,a.devPtr,size()*sizeof(float),cudaMemcpyDeviceToDevice, ::get_cuda_stream());                
        
    }
    ~GPUVector() {        
        return_memory(size(),devPtr);        
        free(host);        
    }


    void init() {                       
        float * temp = find_memory(size());
        if(temp == NULL) {
            cudaError_t err = cudaMalloc((void**)&temp,size()*sizeof(float));                                
            if(err != cudaSuccess) {                
                std::cout << "N=" << N << std::endl;                
                std::cout << cudaGetErrorString(err) << std::endl;
            }
            assert(err == cudaSuccess);          
        }
        // using 25%?        
        host = (float*)calloc(size(),sizeof(float));
        assert(host != NULL);                    
        devPtr = temp;
    }
    void rand(float min, float max) {
        typedef std::chrono::high_resolution_clock myclock;
        myclock::time_point beginning = myclock::now();
        myclock::duration d = myclock::now() - beginning;
        unsigned seed = d.count();
        std::default_random_engine generator(seed);
        std::uniform_real_distribution<double> distribution(min,max);                
        for(size_t i = 0; i < size(); i++) host[i] = distribution(generator);
        cudaMemcpyAsync(devPtr,host,size()*sizeof(float),cudaMemcpyHostToDevice, ::get_cuda_stream());            
    }
    size_t size() const { return N; }


    GPUVector& operator = (const GPUVector & a)   {      
        if(a.devPtr == devPtr) return *this;        
        if(devPtr != nullptr) return_memory(size(),devPtr);
        if(host != nullptr) free(host);
        N = a.N;        
        host  = a.host;
        devPtr = a.devPtr;        
        return *this;
        
    }


    void download_host()     {
        cudaMemcpyAsync(host,devPtr,size()*sizeof(float),cudaMemcpyDeviceToHost, ::get_cuda_stream());
    }
    void upload_device()     {
        cudaMemcpyAsync(devPtr,host,size()*sizeof(float),cudaMemcpyHostToDevice, ::get_cuda_stream());
    }    
    void zero()    {
        //for(size_t i = 0; i < M*N*O*P; i++) host[i] = 0;
        //memset(host,0x0,size()*sizeof(float));
        //cudaMemcpyAsync(devPtr,host,size()*sizeof(float),cudaMemcpyHostToDevice, ::get_cuda_stream());
        cudaMemsetAsync(devPtr,0x00,size()*sizeof(float),::get_cuda_stream());
    }
    void ones()    {
        fill(1.0f);
    }    
    void randu() {
        rand(0.0f,1.0f);
    }
    void random(float min, float max) {
        rand(min,max);
    }
    void fill(const float val)     {
        for(size_t i = 0; i < size(); i++) host[i] = val;
        cudaMemcpyAsync(devPtr,host,size()*sizeof(float),cudaMemcpyHostToDevice, ::get_cuda_stream());
    }

    float *DP()   const  { return this->devPtr; }
    size_t Dim1() const { return this->N; }
    size_t Dim2() const { return 0; }
    size_t Dim3() const { return 0; }
    size_t Dim4() const { return 0; }

    void resize(size_t i) {        
        if(host != nullptr) free(host);        
        if(devPtr != nullptr) return_memory(size(),devPtr);
        N = i;        
        init();        
    }
    bool has_nans() 
    {        
        bool nans = false;
        download_host();
        for(size_t i = 0; i < size(); i++) 
        {
            if(isnan(host[i])) {
                std::cout << "NaN detected" << std::endl;
                return true;
            }
        }
        return false;
    }
    // to use these you must download_host() first.
    // when you are done you must upload_device()
    // if you do not first download it will not be in sync with the device
    // if you do not upload it back to the device it will not be in sync with the device
    float& operator[](array_index pos)    {        
        assert(pos < size());
        while(pos < 0) pos += size();
        return host[pos];        
    }    
    float    __getitem(array_index pos) { while(pos < 0) pos += N; return host[pos]; }
    void     __setitem(array_index pos, float value) { while(pos < 0) pos += N; host[pos] = value; }

        
    
    void print()  {
        download_host();               
        std::cout << "vector[" << N << "]" << std::endl;
        for(size_t w = 0; w < size(); w++) {
            std::cout << host[w] << ",";
        }
        std::cout << std::endl;        
    }

    // vector
    void SetVector(const float * vector, int incx = 0, int incy=0)    {
        cublasStatus_t err = cublasSetVector(N,sizeof(float),vector,incx,devPtr,incy);
        ASSERTOK(err);
    }
    float* GetVector(float * buffer, int incx=0, int incy=0)    {
        cublasStatus_t err = cublasGetVector(N,sizeof(float),devPtr,incx,buffer,incy);
        ASSERTOK(err);
        return buffer;
    }
    void SetVectorAsync(CudaStream & stream, const float * vector, int incx = 0, int incy=0)    {
        cublasStatus_t err = cublasSetVectorAsync(N,sizeof(float),vector,incx,devPtr,incy,stream.stream);
        ASSERTOK(err);
    }
    float* GetVectorAsync(CudaStream & stream,  float * buffer, int incx=0, int incy=0)    {
        cublasStatus_t err = cublasGetVectorAsync(N,sizeof(float),devPtr,incx,buffer,incy,stream.stream);
        ASSERTOK(err);
        return buffer;
    }

    
    // make a new copy
    GPUVector copy() const {
        GPUVector r(N);        
        cudaMemcpyAsync(r.devPtr,devPtr,size()*sizeof(float), cudaMemcpyDeviceToDevice, ::get_cuda_stream());
        return r;
    }
    GPUVector clone() const { return copy(); } 
    GPUVector eval() const { return copy(); }     

    GPUObjectType_t type() const { return GPU_VECTOR; }

    GPUVector operator -() { return GPUVector(*this * -1); }

    GPUVector operator + (GPUVector & y);
    GPUVector operator - (GPUVector & y);
    GPUVector operator * (GPUVector & y);
    GPUVector operator / (GPUVector & y);
    GPUVector operator % (GPUVector & y);
    
    
    GPUVector operator + (float v);
    GPUVector operator - (float v);
    GPUVector operator * (float v);
    GPUVector operator / (float v);
    GPUVector operator % (float v);

    GPUVector operator + (const GPUConst & v);
    GPUVector operator - (const GPUConst & v);
    GPUVector operator * (const GPUConst & v);
    GPUVector operator / (const GPUConst & v);
    GPUVector operator % (const GPUConst & v);

    GPUVector operator + (const GPUScalar & s);
    GPUVector operator - (const GPUScalar & s);
    GPUVector operator * (const GPUScalar & s);
    GPUVector operator / (const GPUScalar & s);
    GPUVector operator % (const GPUScalar & s);

    float dot(const GPUVector & b);
    float mag();
    
};

GPUVector linspaced(float start, float stop, float num_vals) {
    GPUVector r(num_vals);
    float delta = (stop-start)/(num_vals-1.0f);
    for(size_t i = 0; i < stop-start; i++) 
        r.host[i] = start + i*delta;
    r.upload_device();
    return r;
}

GPUVector sequence(size_t start, size_t num, size_t incr) {
    GPUVector r( num * incr );
    for(size_t i = 0; i < num; i+=incr)
        r.host[i] = i + start;
    r.upload_device();
    return r;
}

typedef GPUVector GPUArray;

struct GPUMatrix 
{
    float * devPtr;
    float * host;
    std::shared_ptr<float> host_ptr;
    size_t     M,N;

    GPUObjectType_t type() const { return GPU_MATRIX; }

    void init() {                       
        float * temp = find_memory(size());
        if(temp == NULL) {
            cudaError_t err = cudaMalloc((void**)&temp,size()*sizeof(float));                                
            if(err != cudaSuccess) {
                std::cout << "M=" << M << std::endl;
                std::cout << "N=" << N << std::endl;                
                std::cout << cudaGetErrorString(err) << std::endl;
            }
            assert(err == cudaSuccess);          
        }
        // using 25%?        
        host = (float*)calloc(size(),sizeof(float));
        assert(host != NULL);                    
        devPtr = temp;   
        zero();
    }
    void rand(float min, float max) {
        typedef std::chrono::high_resolution_clock myclock;
        myclock::time_point beginning = myclock::now();
        myclock::duration d = myclock::now() - beginning;
        unsigned seed = d.count();
        std::default_random_engine generator(seed);
        std::uniform_real_distribution<double> distribution(min,max);                
        for(size_t i = 0; i < size(); i++) host[i] = distribution(generator);
        cudaMemcpyAsync(devPtr,host,size()*sizeof(float),cudaMemcpyHostToDevice, ::get_cuda_stream());            
    }

    GPUMatrix() {
        M = N = 0;
        devPtr = nullptr;
        host   = nullptr;
    }
    GPUMatrix(size_t m, size_t n)    {
        assert(m > 0);
        M = m;
        assert(n > 0);
        N = n;                
        init();        
    }
    GPUMatrix(const std::vector<float> & data, size_t m, size_t n)    {
        assert(m > 0);
        M = m;    
        N = n;                        
        init();
        cudaMemcpyAsync(devPtr,data.data(),size()*sizeof(float),cudaMemcpyHostToDevice, ::get_cuda_stream());
    }
    // p is on device
    GPUMatrix(float * devptr, size_t m, size_t n, size_t x=0,size_t y=0)   {
        assert(m > 0);
        assert(n > 0);
        M = m;    
        N = n;                
        host = (float*)calloc(size(),sizeof(float));                
        devPtr = devptr;        
    }
    GPUMatrix(const GPUMatrix & a) {        
        M = a.M;        
        N = a.N;        
        init();
        cudaMemcpyAsync(devPtr,a.devPtr,size()*sizeof(float),cudaMemcpyDeviceToDevice, ::get_cuda_stream());                
        
    }
    ~GPUMatrix() {        
        return_memory(size(),devPtr);        
        free(host);        
    }

    size_t size() const { return M*N; }

    void resize(size_t i, size_t j) {        
        if(host != nullptr) free(host);        
        if(devPtr != nullptr) return_memory(size(),devPtr);
        M = i;
        N = j;
        init();        
    }

    void copy_meta(const GPUMatrix & c) {
        if(c.devPtr == NULL || c.M != M || c.N != N) 
        {        
            if(host)   free(host);
            if(devPtr) return_memory(size(),devPtr);
            M = c.M;
            N = c.N;                        
            init();
        }
    }
    
    GPUMatrix& operator = (const GPUMatrix & a)   {      
        if(a.devPtr == devPtr) return *this;        
        if(devPtr != nullptr) return_memory(size(),devPtr);
        if(host != nullptr) free(host);
        M = a.M;
        N = a.N;            
        host  = a.host;
        devPtr = a.devPtr;        
        return *this;        
    }

    void download_host()     {
        cudaMemcpyAsync(host,devPtr,size()*sizeof(float),cudaMemcpyDeviceToHost, ::get_cuda_stream());
    }
    void upload_device()     {
        cudaMemcpyAsync(devPtr,host,size()*sizeof(float),cudaMemcpyHostToDevice, ::get_cuda_stream());
    }
    void zero()    {
        //for(size_t i = 0; i < M*N*O*P; i++) host[i] = 0;
        //memset(host,0x0,size()*sizeof(float));
        //cudaMemcpyAsync(devPtr,host,size()*sizeof(float),cudaMemcpyHostToDevice, ::get_cuda_stream());
        cudaMemsetAsync(devPtr,0x00,size()*sizeof(float),::get_cuda_stream());
    }
    void ones()    {
        fill(1.0f);
    }    
    void randu() {
        rand(0.0f,1.0f);
    }
    void random(float min, float max) {
        rand(min,max);
    }
    void fill(const float val)     {
        for(size_t i = 0; i < size(); i++) host[i] = val;
        cudaMemcpyAsync(devPtr,host,size()*sizeof(float),cudaMemcpyHostToDevice, ::get_cuda_stream());
    }

    float *DP()   const  { return this->devPtr; }
    size_t Dim1() const { return this->M; }
    size_t Dim2() const { return this->N; }
    size_t Dim3() const { return 0; }
    size_t Dim4() const { return 0; }

    void resize(size_t i) {        
        if(host != nullptr) free(host);        
        if(devPtr != nullptr) return_memory(size(),devPtr);
        N = i;        
        init();        
    }
    bool has_nans() 
    {        
        bool nans = false;
        download_host();
        for(size_t i = 0; i < size(); i++) 
        {
            if(isnan(host[i])) {
                std::cout << "NaN detected" << std::endl;
                return true;
            }
        }
        return false;
    }
    // to use these you must download_host() first.
    // when you are done you must upload_device()
    // if you do not first download it will not be in sync with the device
    // if you do not upload it back to the device it will not be in sync with the device
    GPUVector    __getitem(array_index pos) { 
        while(pos < 0) pos += N; 
        GPUVector v(N);
        cudaMemcpyAsync(v.devPtr,devPtr+pos*N, N*sizeof(float),cudaMemcpyDeviceToDevice,get_cuda_stream());
        return v;        
    }
    void     __setitem(array_index pos, GPUVector value) 
    { 
        while(pos < 0) pos += N; 
        cudaMemcpyAsync(devPtr+pos*N,value.devPtr,N*sizeof(float),cudaMemcpyDeviceToDevice,get_cuda_stream());
    }

// same thing but built into cublas    
    void SetMatrix(float * matrix)    {
        int lda = M;
        int ldb = M;
        cublasStatus_t err = cublasSetMatrix(M,N,size()*sizeof(float),matrix,lda,devPtr,ldb);
        ASSERTOK(err);
    }    
    float* GetMatrix(float * buffer)    {     
        int lda = M;
        int ldb = M;
        cublasStatus_t err = cublasGetMatrix(M,N,size()*sizeof(float),devPtr,lda,buffer,ldb);
        ASSERTOK(err);
        return buffer;
    }    
    void SetMatrixAsync(float * matrix)    {
        int lda = M;
        int ldb = M;
        cublasStatus_t err = cublasSetMatrixAsync(M,N,sizeof(float),matrix,lda,devPtr,ldb,::get_cuda_stream());
        ASSERTOK(err);
    }
    float* GetMatrixAsync(float * buffer)    {     
        int lda = M;
        int ldb = M;
        cublasStatus_t err = cublasGetMatrixAsync(M,N,sizeof(float),devPtr,lda,buffer,ldb,::get_cuda_stream());
        ASSERTOK(err);
        return buffer;
    }

    // make a new copy
    GPUMatrix copy() const {
        // if vector memcpy1d
        // if matrix memcpy2d
        // if cube   memcpy3d 
        // fields can not be copied
        
        GPUMatrix r(M,N);        
       cudaMemcpyAsync(r.devPtr,devPtr,size()*sizeof(float), cudaMemcpyDeviceToDevice, ::get_cuda_stream());
        return r;
    }
    GPUMatrix clone() const { return copy(); } 
    GPUMatrix eval() const { return copy(); }     

    GPUMatrix operator - () { return GPUMatrix(*this * -1.0f); }    
    GPUMatrix operator + (const GPUMatrix & m);
    GPUMatrix operator - (const GPUMatrix & m);    
    GPUMatrix operator * (const GPUMatrix & m);
 
    // not standard linear algebra operators
    GPUMatrix operator / (const GPUMatrix & m) { return GPUMatrix(*this / m); }
    GPUMatrix operator % (const GPUMatrix & m) { return GPUMatrix(*this % m); }

    
    GPUMatrix operator + (float m) { return GPUMatrix(*this + m); }    
    GPUMatrix operator - (float m) { return GPUMatrix(*this - m); }    
    GPUMatrix operator * (float m) { return GPUMatrix(*this * m); }    
    GPUMatrix operator / (float m) { return GPUMatrix(*this / m); }
    GPUMatrix operator % (float m) { return GPUMatrix(*this % m); }



//////////////////////
/// rowwise
//////////////////////

    GPUMatrix operator * (const GPUVector & v)    {
        assert(Dim2() == v.N);
        GPUMatrix r(*this);        
        for(size_t i = 0; i < M; i++)        {
            vector_mulf_row(r.devPtr,(i*N),v.devPtr,0,v.N);
        }                
        return r;
    }    
    GPUMatrix operator + (const GPUVector & v)    {
        assert(Dim2() == v.N);
        GPUMatrix r(*this);               
        for(size_t i = 0; i < M; i++)        {            
            vector_addf_row(r.devPtr,(i*N),v.devPtr,0,v.N);
        }        
        return r;
    }
    GPUMatrix operator - (const GPUVector & v)    {
        assert(Dim2() == v.N);
        GPUMatrix r(*this);               
        for(size_t i = 0; i < M; i++)        {
            vector_subf_row(r.devPtr,(i*N),v.devPtr,0,v.N);
        }
        return r;
    }
    GPUMatrix operator / (const GPUVector & v)    {
        assert(Dim2() == v.N);
        GPUMatrix r(*this);               
        for(size_t i = 0; i < M; i++)        {
            vector_divf_row(r.devPtr,(i*N),v.devPtr,0,v.N);
        }
        return r;
    }
    GPUMatrix operator % (const GPUVector & v)    {
        assert(Dim2() == v.N);
        GPUMatrix r(*this);               
        for(size_t i = 0; i < M; i++)        {
            vector_modf_row(r.devPtr,(i*N),v.devPtr,0,v.N);
        }
        return r;
    }

    GPUMatrix operator + (const GPUConst & v);
    GPUMatrix operator - (const GPUConst & v);
    GPUMatrix operator * (const GPUConst & v);
    // this is not valid linear algebra
    GPUMatrix operator / (const GPUConst & v);    
    GPUMatrix operator % (const GPUConst & v);

    GPUMatrix operator + (const GPUScalar & s);
    GPUMatrix operator - (const GPUScalar & s);
    GPUMatrix operator * (const GPUScalar & s);
    // this is not valid linear algebra
    GPUMatrix operator / (const GPUScalar & s);
    GPUMatrix operator % (const GPUScalar & s);


//////////////////////////////////
// rowwise matrix is a big mess
//////////////////////////////////


    GPUVector row(array_index row) {
        if(row < 0) row += Dim1();
        GPUVector r(Dim2());
        for(size_t i = 0; i < Dim2(); i++)
            r[i] = get(row,i);
        return r;
    }
    GPUVector col(array_index col) {
        if(col < 0) col += Dim2();
        GPUVector r(Dim1());
        for(size_t i = 0; i < Dim1(); i++)
            r[i] = get(i,col);
        return r;
    }
    
    void addToEachRow(const GPUMatrix &b, int row=0) {
        assert(Dim2() == b.Dim2());
        for(size_t i = 0; i < M; i++)
            vector_addf_row(devPtr,(i*N),b.devPtr,row,N);        
    }

    // row selects row in m to add to each row in matrix
    GPUMatrix row_add(const GPUMatrix & m, int row=0) {
        assert(Dim2() == m.Dim2());
        GPUMatrix r(*this);        
        for(size_t i = 0; i < M; i++)
            vector_addf_row(r.devPtr,(i*N),m.devPtr,row,N);        
        return r;
    }
    GPUMatrix row_sub(const GPUMatrix & m, int row=0) {
        GPUMatrix r(*this);        
        for(size_t i = 0; i < M; i++)
            vector_subf_row(r.devPtr,(i*N),m.devPtr,row,N);        
        return r;
    }
    GPUMatrix row_mul(const GPUMatrix & m, int row=0) {
        GPUMatrix r(*this);        
        for(size_t i = 0; i < M; i++)
            vector_mulf_row(r.devPtr,(i*N),m.devPtr,row,N);        
        return r;
    }    
    GPUMatrix row_div(const GPUMatrix & m, int row=0) {
        GPUMatrix r(*this);        
        for(size_t i = 0; i < M; i++)
            vector_divf_row(r.devPtr,(i*N),m.devPtr,row,N);        
        return r;
    }
    GPUMatrix row_mod(const GPUMatrix & m, int row=0) {
        GPUMatrix r(*this);        
        for(size_t i = 0; i < M; i++)
            vector_modf_row(r.devPtr,(i*N),m.devPtr,row,N);        
        return r;
    }

    GPUMatrix get_row(int row) {
        GPUMatrix r(1,cols());
        cudaMemcpyAsync(r.devPtr,devPtr+row*N,cols()*sizeof(float),cudaMemcpyDeviceToDevice);
        return r;
    }
    void set_row(const GPUMatrix & m, int dst_row=0, int src_row=0) {
        assert(N == m.N);
        //vector_setrowf(devPtr, dst_row, m.devPtr, src_row, N);
        cudaMemcpyAsync(devPtr + dst_row*N, m.devPtr + src_row*m.N, m.N*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    void set_row(const GPUVector & v, int dst_row=0) {
        assert(N == v.N);
        //vector_setrowf(devPtr, dst_row, v.devPtr, 0, N);
        cudaMemcpyAsync(devPtr + dst_row*N, v.devPtr, v.N*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    

//////////////////////
// matmul
//////////////////////
    GPUMatrix matmul(const GPUMatrix & b, bool transa=false,bool transb=false, bool transc=false,float alpha=1.0,float beta=0.0);
    GPUMatrix mm(const GPUMatrix & b, bool transa=false,bool transb=false,bool transc = false,float alpha=1.0,float beta=0.0) { return matmul(b,transa,transb,transc,alpha,beta); }
    GPUVector matvec(const GPUVector & v, bool transa=false, float alpha = 1.0, float beta = 0.0);
    GPUVector mv(const GPUVector & v, bool transa=false,float alpha=1.0,float beta=0.0) { return matvec(v,transa,alpha,beta); }


    float& operator()(array_index i, array_index j) {
        if(i < 0) i += Dim1();
        if(j < 0) j += Dim2();        
        return host[i*Dim2()+j];
    }
    float& operator[](array_index index) { 
        if(index < 0) index += Dim1()*Dim2();
        return host[index]; 
    }
    
    float get(array_index r, array_index c) {         
        if(r < 0) r += Dim1();
        if(c < 0) c += Dim2();
        return host[get_index(r,c)];
    }    
    void set(array_index r, array_index c, float value) {
        if(r < 0) r += Dim1();
        if(c < 0) c += Dim2();
        host[get_index(r,c)] = value;        
    }

    size_t get_index(array_index r, array_index c) { 
        if(r < 0) r += Dim1();
        if(c < 0) c += Dim2();    
        return get_index(r,c); 
    }
    void   set_index(array_index r, array_index c, float value) { 
        if(r < 0) r += Dim1();
        if(c < 0) c += Dim2();    
        set_index(r,c,value); 
    }
    
    size_t rows() { return M; }
    size_t cols() { return N; }

    // is it called rank?
    void swap_order() {
        array_index temp = M;
        M = N;
        N = temp;
    }

    void print() {
        //print();
        download_host();        
        for(size_t i = 0; i < Dim1(); i++) {
            for(size_t j = 0; j < Dim2(); j++) 
            {
                std::cout << (*this)(i,j);
                if(j < (Dim2()-1)) std::cout << ",";
            }            
            std::cout << std::endl;
        }        
    }
    void print_dims() const {
        std::cout << "Matrix(" << Dim1() << "," << Dim2() << ")" << std::endl;
    }

    void identity()  {     
        // identity only makes sense on square matrix.   
        assert(Dim1() == Dim2());
        size_t c = 0;
        download_host();
        fill(0);
        for(size_t i = 0; i < M; i++) {
            host[i*N + c++] = 1;
        }            
        upload_device();
    }

    GPUMatrix t();

};







///////////////////////////////////////////////////////////////////////
// vector maths
////////////////////////////////////////////////////////////////////////

// this is done on CPU at the moment.


float sum(const GPUVector & a)     {
    float p = vector_sumf(a.devPtr,a.size());
    return p;
}
float prod(const GPUVector & a)     {
    float p = vector_prodf(a.devPtr,a.size());
    return p;
}

float sum(const GPUMatrix & a)     {
    float p = vector_sumf(a.devPtr,a.size());
    return p;
}
float prod(const GPUMatrix & a)     {
    float p = vector_prodf(a.devPtr,a.size());
    return p;
}




///////////////////////////////////////////////////////////////////////
// cuda math
///////////////////////////////////////////////////////////////////////

template<typename T>
T add(const T & a, const T& b)     {    
    /// array can solve them all
    float * p = array_addf(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T add(const T & a, const float b)     {    
    /// array can solve them all
    float * p = array_addf_const(a.DP(),b,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T add(const T & a, const Scalar& b)     {    
    /// array can solve them all
    float * p = array_addf_const(a.DP(),b.value,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T add(const T & a, const GPUConst& b)     {    
    /// array can solve them all
    float * p = array_addf_scalar(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T add(const T & a, const GPUScalar& b)     {    
    /// array can solve them all
    float * p = array_addf_scalar(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T sub(const T & a, const T& b)     {    
    /// array can solve them all
    float * p = array_subf(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T sub(const T & a, const float b)     {    
    /// array can solve them all
    float * p = array_subf_const(a.DP(),b,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T sub(const T & a, const Scalar& b)     {    
    /// array can solve them all
    float * p = array_subf_const(a.DP(),b.value,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T sub(const T & a, const GPUConst& b)     {    
    /// array can solve them all
    float * p = array_subf_scalar(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T sub(const T & a, const GPUScalar& b)     {    
    /// array can solve them all
    float * p = array_subf_scalar(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T mul(const T & a, const T& b)     {    
    /// array can solve them all
    float * p = array_mulf(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T mul(const T & a, const float b)     {    
    /// array can solve them all
    float * p = array_mulf_const(a.DP(),b,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T mul(const T & a, const Scalar& b)     {    
    /// array can solve them all
    float * p = array_mulf_const(a.DP(),b.value,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T div(const T & a, const T& b)     {    
    /// array can solve them all
    float * p = array_divf(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T div(const T & a, const float b)     {    
    /// array can solve them all
    float * p = array_divf_const(a.DP(),b,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T div(const T & a, const Scalar& b)     {    
    /// array can solve them all
    float * p = array_divf_const(a.DP(),b.value,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T div(const T & a, const GPUConst& b)     {    
    /// array can solve them all
    float * p = array_divf_scalar(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T div(const T & a, const GPUScalar& b)     {    
    /// array can solve them all
    float * p = array_divf_scalar(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T mod(const T & a, const T& b)     {    
    /// array can solve them all
    float * p = array_modf(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T mod(const T & a, const float b)     {    
    /// array can solve them all
    float * p = array_modf_const(a.DP(),b,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T mod(const T & a, const Scalar& b)     {    
    /// array can solve them all
    float * p = array_modf_const(a.DP(),b.value,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T mod(const T & a, const GPUConst& b)     {    
    /// array can solve them all
    float * p = array_modf_scalar(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T mod(const T & a, const GPUScalar& b)     {    
    /// array can solve them all
    float * p = array_modf_scalar(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}






///////////////////////////////////////////////////////////////////////
// cuda math functions
///////////////////////////////////////////////////////////////////////

template<typename T>
T acos(const T & a)     {        
    float * p = array_acosf(a.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T acosh(const T & a)     {
    float * p = array_acoshf(a.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T asin(const T & a)     {
    float * p = array_asinf(a.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T asinh(const T & a)     {
    float * p = array_asinhf(a.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T atan2(const T & a,const T & b)     {
    float * p = array_atan2f(a.DP(),b.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T atan2(const T& a,const float b)     {    
    float * p = array_atan2f_const(a.DP(),b, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T atan2(const T& a,const Scalar& b)     {    
    float * p = array_atan2f_const(a.DP(),b.value, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T atan2(const T& a,const GPUConst& b)     {    
    float * p = array_atan2f_scalar(a.DP(),b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T atan2(const T& a,const GPUScalar& b)     {    
    float * p = array_atan2f_scalar(a.DP(),b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T atan(const T & a)     {
    float * p = array_atanf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T atanh(const T & a)     {
    float * p = array_atanhf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T ceil(const T& a)     {
    float * p = array_ceilf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T cbrt(const T& a)     {
    float * p = array_cbrtf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T rcbrt(const T& a)     {
    float * p = array_rcbrtf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T cos(const T & a)     {
    float * p = array_cosf(a.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T cosh(const T & a)     {
    float * p = array_coshf(a.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T cospi(const T & a)     {
    float * p = array_cospif(a.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T cyl_bessel_i0(const T& a)     {
    float * p = array_cyl_bessel_i0f(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T cyl_bessel_i1(const T& a)     {
    float * p = array_cyl_bessel_i1f(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}



template<typename T>
T exp10(const T & a)     {
    float * p = array_exp10f(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T exp2(const T & a)     {
    float * p = array_exp2f(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T exp(const T & a)     {
    float * p = array_expf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T expm1(const T & a)     {
    float * p = array_expm1f(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T fabs(const T & a)     {
    float * p = array_fabsf(a.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T floor(const T & a)     {
    float * p = array_floorf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}



template<typename T>
T fmax(const T & a,const T & b)     {
    float * p = array_fmaxf(a.DP(),b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fmax(const T & a,const float b)     {
    float * p = array_fmaxf_const(a.DP(),b, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fmax(const T & a, const Scalar& b)     {
    float * p = array_fmaxf_const(a.DP(),b.value, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fmax(const T & a, const GPUScalar & b)     {
    float * p = array_fmaxf_scalar(a.DP(),b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fmax(const T & a, const GPUConst & b)     {
    float * p = array_fmaxf_scalar(a.DP(),b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T fmin(const T & a,const T & b)     {
    float * p = array_fminf(a.DP(),b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fminf(const T & a,const float b)     {
    float * p = array_fmaxf_const(a.DP(),b, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fminf(const T & a,const Scalar& b)     {
    float * p = array_fmaxf_const(a.DP(),b.value, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fminf(const T & a, const GPUScalar& b)     {
    float * p = array_fmaxf_const(a.DP(),b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fminf(const T & a, const GPUConst& b)     {
    float * p = array_fmaxf_const(a.DP(), b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}



template<typename T>
T fmod(const T& a,const T& b)     {
    float * p = array_fmodf(a.DP(),b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fmod(const T& a,const float b)     {
    float * p = array_fmodf_const(a.DP(),b, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fmod(const T& a, const Scalar& b)     {
    float * p = array_fmodf_const(a.DP(),b.value, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fmod(const T& a, const GPUScalar& b)     {
    float * p = array_fmodf_scalar(a.DP(),b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fmod(const T& a, const GPUConst& b)     {
    float * p = array_fmodf_scalar(a.DP(),b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T log10(const T & a) {
    float * p = array_log10f(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T log1p(const T & a) {
    float * p = array_log1pf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T log2(const T & a) {
    float * p = array_log2f(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T logb(const T & a) {
    float * p = array_logbf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T pow(const T & a,const T & b) {
    float * p = array_powf(a.DP(), b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T pow(const T& a,const float  b) {    
    float * p = array_powf_const(a.DP(), b, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T pow(const T& a,const Scalar&  b) {    
    float * p = array_powf_const(a.DP(), b.value, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T pow(const T& a,const GPUScalar&  b) {    
    float * p = array_powf_scalar(a.DP(), b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T pow(const T& a,const GPUConst&  b) {    
    float * p = array_powf_scalar(a.DP(), b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T rsqrt(const T & a) {
    float * p = array_rsqrtf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T sin(const T& a) {
    float * p = array_sinf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T sinh(const T & a) {
    float * p = array_sinhf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T sinpi(const T& a) {
    float * p = array_sinpif(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T sqrt(const T& a) {
    float * p = array_sqrtf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T tan(const T& a) {
    float * p = array_tanf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T tanh(const T& a) {    
    float * p = array_tanhf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T sigmoid(const T& a)
{    
    float * p = array_sigmoidf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T sigmoid_grad(const T& a)
{    
    float * p = array_sigmoid_gradf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T tanh_grad(const T& a)
{    
    float * p = array_tanh_gradf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T relu(const T& a)
{    
    float * p = array_reluf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T relu_grad(const T & a)
{    
    float * p = array_relu_gradf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T softmax(const T& a)
{    
    float * p = array_softmaxf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
}

// todo constant
template<typename T>
T fdivide(const T & a,const T & b)     {
    float * p = array_fdimf(a.DP(), b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fdivide(const T & a,const float b)     {
    float * p = array_fdimf_const(a.DP(), b, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fdivide(const T & a,const Scalar & b)     {
    float * p = array_fdimf_const(a.DP(), b.value, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fdivide(const T & a,const GPUConst & b)     {
    float * p = array_fdimf_scalar(a.DP(), b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fdivide(const T & a,const GPUScalar & b)     {
    float * p = array_fdimf_scalar(a.DP(), b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}




template<typename T>
T fma(const T & a,const T & b, const T &c)     {
    float * p = array_fmaf(a.DP(), b.DP(), c.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T hypot(const T & a,const T& b)     {
    float * p = array_hypotf(a.DP(), b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T hypot(const T & a,const float b)     {
    float * p = array_hypotf_const(a.DP(), b, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T hypot(const T & a,const Scalar& b)     {
    float * p = array_hypotf_const(a.DP(), b.value, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T hypot(const T & a,const GPUConst& b)     {
    float * p = array_hypotf_scalar(a.DP(), b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T hypot(const T & a,const GPUScalar& b)     {
    float * p = array_hypotf_scalar(a.DP(), b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T rhypot(const T & a,const T& b)     {
    float * p = array_rhypotf(a.DP(), b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T rhypot(const T & a,const float b)     {
    float * p = array_rhypotf_const(a.DP(), b, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T rhypot(const T & a,const Scalar& b)     {
    float * p = array_rhypotf_const(a.DP(), b.value, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T rhypot(const T & a,const GPUConst& b)     {
    float * p = array_rhypotf_scalar(a.DP(), b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T rhypot(const T & a,const GPUScalar& b)     {
    float * p = array_rhypotf_scalar(a.DP(), b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T trunc(const T & a)     {
    float * p = array_truncf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T ilogb(const T & a)     {
    float * p = array_ilogbf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T j0(const T& a)     {
    float * p = array_j0f(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T j1(const T & a)     {
    float * p = array_j1f(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T jn(const T& a,int N)     {
    float * p = array_jnf(a.DP(),N, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T ldexp(const T& a,int exp) {
    float * p = array_ldexpf(a.DP(),exp, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T lgamma(const T& a) {
    float * p = array_lgammaf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

// todo constant
template<typename T>
T norm3(const T & a, const T& b, const T& c) {
    float * p = array_norm3df(a.DP(), b.DP(),c.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T rnorm3(const T & a, const T& b, const T& c) {
    float * p = array_rnorm3df(a.DP(), b.DP(),c.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

// todo constant
template<typename T>
T norm4(const T& a,const  T& b, const T &c, const T& d) {
    float * p = array_norm4df(a.DP(), b.DP(), c.DP(), d.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T rnorm4(const T& a,const  T& b, const T &c, const T& d) {
    float * p = array_rnorm4df(a.DP(), b.DP(), c.DP(), d.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T normcdf(const T& a) {
    float * p = array_normcdff(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T normcdfinv(const T& a) {
    float * p = array_normcdfinvf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T norm(int dim, const T& a) {
    float * p = array_normf(dim,a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T rnorm(int dim, const T& a) {
    float * p = array_rnormf(dim,a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T scalbln(const T& a, long int N) {
    float * p = array_scalblnf(a.DP(), N, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T tgamma(const T& a) {
    float * p = array_tgammaf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T y0(const T& a) {
    float * p = array_y0f(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T y1(const T& a) {
    float * p = array_y1f(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T yn(int N,const T& a) {
    float * p = array_ynf(N,a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}    


template<typename T>
T copysign(const T& a,const T& b)     {
    float * p = array_copysignf(a.DP(), b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T erfc(const T & a)     {
    float * p = array_erfcf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T erfcinv(const T& a)     {
    float * p = array_erfinvf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T erfcx(const T& a)     {
    float * p = array_erfcxf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T erf(const T& a)     {
    float * p = array_erff(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T erfinv(const T& a)     {
    float * p = array_erff(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T fdim(const T& a,const T& b)     {    
    float * p = array_fdimf(a.DP(), b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fdim(const T& a,const float b)     {    
    float * p = array_fdimf_const(a.DP(), b, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fdim(const T& a,const Scalar& b)     {    
    float * p = array_fdimf_const(a.DP(), b.value, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fdim(const T& a,const GPUScalar& b)     {    
    float * p = array_fdimf_scalar(a.DP(), b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T fdim(const T& a,const GPUConst& b)     {    
    float * p = array_fdimf_scalar(a.DP(), b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}


template<typename T>
T nearbyint(const T & a) {
    float * p = array_nearbyintf(a.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}

template<typename T>
T remainder(const T & a,const T & b) {
    float * p = array_remainderf(a.DP(), b.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T remainder(const T & a,const float b) {    
    float * p = array_remainderf_const(a.DP(),b, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T remainder(const T & a,const Scalar& b) {    
    float * p = array_remainderf_const(a.DP(),b.value, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T remainder(const T & a,const GPUConst& b) {    
    float * p = array_remainderf_scalar(a.DP(),b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}
template<typename T>
T remainder(const T & a,const GPUScalar& b) {    
    float * p = array_remainderf_scalar(a.DP(),b.devPtr, a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return T(p,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());
}



/////////////////////////////////////////////////////////
// virtual machine 
/////////////////////////////////////////////////////////

template<typename T>
T& reg_add(const T & a, const T& b, T& r)     {    
    //assert(a == b);
    /// array can solve them all
    r.copy_meta(a);    
    array_r_addf(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_add(const T & a, const float b, T& r)     {    
    //assert(a == b);
    /// array can solve them all
    r.copy_meta(a);    
    array_r_addf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_add(const T & a, const Scalar& b, T& r)     {    
    //assert(a == b);
    /// array can solve them all
    r.copy_meta(a);    
    array_r_addf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_add(const T & a, const GPUConst& b, T& r)     {    
    //assert(a == b);
    /// array can solve them all
    r.copy_meta(a);    
    array_r_addf_scalar(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_add(const T & a, const GPUScalar& b, T& r)     {    
    //assert(a == b);
    /// array can solve them all
    r.copy_meta(a);    
    array_r_addf_scalar(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}

template<typename T>
T& reg_sub(const T & a, const T& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_subf(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_sub(const T & a, const float b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_subf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_sub(const T & a, const Scalar& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_subf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_sub(const T & a, const GPUConst& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_subf_scalar(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_sub(const T & a, const GPUScalar& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_subf_scalar(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}

template<typename T>
T& reg_mul(const T & a, const T& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_mulf(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_mul(const T & a, const float b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_mulf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_mul(const T & a, const Scalar& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_mulf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_mul(const T & a, const GPUConst& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_mulf_scalar(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_mul(const T & a, const GPUScalar& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_mulf_scalar(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}

template<typename T>
T& reg_div(const T & a, const T& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_divf(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_div(const T & a, const float b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_divf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_div(const T & a, const Scalar& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_divf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_div(const T & a, const GPUConst& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_divf_scalar(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_div(const T & a, const GPUScalar& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_divf_scalar(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}

template<typename T>
T& reg_mod(const T & a, const T& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_modf(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_mod(const T & a, const float b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_modf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_mod(const T & a, const Scalar& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_modf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_mod(const T & a, const GPUConst& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_modf_scalar(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_mod(const T & a, const GPUScalar& b, T& r)     {    
    /// array can solve them all
    r.copy_meta(a);    
    array_r_modf_scalar(a.DP(),b.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}



template<typename T>
T& reg_acos(const T& a, T& r)     {        
    r.copy_meta(a);    
    array_r_acosf(a.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_acosh(const T & a, T & r)     {
    r.copy_meta(a);    
    array_r_acoshf(a.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_asin(const T & a, T & r)     {
    r.copy_meta(a);    
    array_r_asinf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_asinh(const T & a, T & r)     {
    r.copy_meta(a);    
    array_r_asinhf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_atan2(const T & a,const T & b, T &r)     {
    r.copy_meta(a);    
    array_r_atan2f(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_atan2(const T& a,const float b, T& r)     {    
    r.copy_meta(a);    
    array_r_atan2f_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_atan2(const T& a,const Scalar& b, T& r)     {    
    r.copy_meta(a);    
    array_r_atan2f_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_atan2(const T & a,const GPUConst & b, T &r)     {
    r.copy_meta(a);    
    array_r_atan2f_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_atan2(const T & a,const GPUScalar & b, T &r)     {
    r.copy_meta(a);    
    array_r_atan2f_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T reg_scalbln(const T& a, long int N, T& r) {
    r.copy_meta(a);    
    array_r_scalblnf(a.DP(), N, r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}


template<typename T>
T& reg_atan(const T& a, T & r)     {
    r.copy_meta(a);    
    array_r_atanf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_atanh(const T& a, T & r)     {
    r.copy_meta(a);    
    array_r_atanhf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_ceil(const T& a, T& r)     {
    r.copy_meta(a);    
    array_r_ceilf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}

template<typename T>
T& reg_cbrt(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_cbrtf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_rcbrt(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_rcbrtf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}

template<typename T>
T& reg_cos(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_cosf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_cosh(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_coshf(a.DP(), r.DP(), a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_cospi(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_cospif(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}

template<typename T>
T& reg_cyl_bessel_i0(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_cyl_bessel_i0f(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_cyl_bessel_i1(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_cyl_bessel_i1f(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}

template<typename T>
T& reg_exp10(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_exp10f(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_exp2(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_exp2f(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_expm1(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_expm1f(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}
template<typename T>
T& reg_exp(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_expf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}


template<typename T>
T& reg_fabs(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_fabsf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}

template<typename T>
T& reg_floor(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_floorf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}



template<typename T>
T& reg_fmax(const T & a,const T & b, T &r)     {
    r.copy_meta(a);    
    array_r_fmaxf(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_fmax(const T& a,const float b, T& r)     {    
    r.copy_meta(a);    
    array_r_fmaxf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_fmax(const T& a,const Scalar& b, T& r)     {    
    r.copy_meta(a);    
    array_r_fmaxf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_fmax(const T & a,const GPUConst & b, T &r)     {
    r.copy_meta(a);    
    array_r_fmaxf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_fmax(const T & a,const GPUScalar & b, T &r)     {
    r.copy_meta(a);    
    array_r_fmaxf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_fmin(const T & a,const T & b, T &r)     {
    r.copy_meta(a);    
    array_r_fminf(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_fmin(const T& a,const float b, T& r)     {    
    r.copy_meta(a);    
    array_r_fminf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_fmin(const T& a,const Scalar& b, T& r)     {    
    r.copy_meta(a);    
    array_r_fminf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_fmin(const T & a,const GPUConst & b, T &r)     {
    r.copy_meta(a);    
    array_r_fminf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_fmin(const T & a,const GPUScalar & b, T &r)     {
    r.copy_meta(a);    
    array_r_fminf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}


template<typename T>
T& reg_fmod(const T & a,const T & b, T &r)     {
    r.copy_meta(a);    
    array_r_fmodf(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_fmod(const T& a,const float b, T& r)     {    
    r.copy_meta(a);    
    array_r_fmodf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_fmod(const T& a,const Scalar& b, T& r)     {    
    r.copy_meta(a);    
    array_r_fmodf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_fmod(const T & a,const GPUConst & b, T &r)     {
    r.copy_meta(a);    
    array_r_fmodf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_fmod(const T & a,const GPUScalar & b, T &r)     {
    r.copy_meta(a);    
    array_r_fmodf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}


template<typename T>
T& reg_log10(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_log10f(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_log2(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_log2f(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_log1p(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_log1pf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_logb(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_logbf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}


template<typename T>
T& reg_pow(const T & a,const T & b, T &r)     {
    r.copy_meta(a);    
    array_r_powf(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_pow(const T& a,const float b, T& r)     {    
    r.copy_meta(a);    
    array_r_powf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_pow(const T& a,const Scalar& b, T& r)     {    
    r.copy_meta(a);    
    array_r_powf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_pow(const T & a,const GPUConst & b, T &r)     {
    r.copy_meta(a);    
    array_r_powf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_pow(const T & a,const GPUScalar & b, T &r)     {
    r.copy_meta(a);    
    array_r_powf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_rsqrt(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_rsqrtf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_sqrt(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_sqrtf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_sin(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_sinf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_sinh(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_sinhf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_sinpi(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_sinpif(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;    
}

template<typename T>
T& reg_tan(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_tanf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_tanh(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_tanhf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_sigmoid(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_sigmoidf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_sigmoid_grad(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_sigmoid_gradf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_relu(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_reluf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_relu_grad(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_relu_gradf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_tanh_grad(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_tanh_gradf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_softmax(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_softmaxf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_trunc(const T & a, T& r)     {
    r.copy_meta(a);    
    array_r_truncf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());    
    return r;
}


template<typename T>
T& reg_fdivide(const T & a,const T & b, T &r)     {
    r.copy_meta(a);    
    array_r_fdividef(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_fdivide(const T& a,const float b, T& r)     {    
    r.copy_meta(a);    
    array_r_fdividef_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_fdivide(const T& a,const Scalar& b, T& r)     {    
    r.copy_meta(a);    
    array_r_fdividef_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_fdivide(const T & a,const GPUConst & b, T &r)     {
    r.copy_meta(a);    
    array_r_fdividef_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_fdivide(const T & a,const GPUScalar & b, T &r)     {
    r.copy_meta(a);    
    array_r_fdividef_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_fma(const T & a,const T & b, const T& c, T &r)     {
    r.copy_meta(a);    
    array_r_fmaf(a.DP(),b.DP(),c.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_hypot(const T & a,const T & b, T &r)     {
    r.copy_meta(a);    
    array_r_hypotf(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_hypot(const T& a,const float b, T& r)     {    
    r.copy_meta(a);    
    array_r_hypotf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_hypot(const T& a,const Scalar& b, T& r)     {    
    r.copy_meta(a);    
    array_r_hypotf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_hypot(const T & a,const GPUConst & b, T &r)     {
    r.copy_meta(a);    
    array_r_hypotf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_hypot(const T & a,const GPUScalar & b, T &r)     {
    r.copy_meta(a);    
    array_r_hypotf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_rhypot(const T & a,const T & b, T &r)     {
    r.copy_meta(a);    
    array_r_rhypotf(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_rhypot(const T& a,const float b, T& r)     {    
    r.copy_meta(a);    
    array_r_rhypotf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_rhypot(const T& a,const Scalar& b, T& r)     {    
    r.copy_meta(a);    
    array_r_rhypotf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_rhypot(const T & a,const GPUConst & b, T &r)     {
    r.copy_meta(a);    
    array_r_rhypotf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_rhypot(const T & a,const GPUScalar & b, T &r)     {
    r.copy_meta(a);    
    array_r_rhypotf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_ilogb(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_ilogbf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_j0(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_j0f(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_j1(const T & a,T &r)     {
    r.copy_meta(a);    
    array_r_j1f(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_jn(const T & a,int n, T &r)     {
    r.copy_meta(a);    
    array_r_jnf(a.DP(),r.DP(),n,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_ldexp(const T & a,int n, T &r)     {
    r.copy_meta(a);    
    array_r_ldexpf(a.DP(),r.DP(),n,a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_lgamma(const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_lgammaf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}


template<typename T>
T& reg_norm3(const T & a,const T & b, const T& c, T &r)     {
    r.copy_meta(a);    
    array_r_norm3df(a.DP(),b.DP(),c.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_rnorm3(const T & a,const T & b, const T& c, T &r)     {
    r.copy_meta(a);    
    array_r_rnorm3df(a.DP(),b.DP(),c.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_norm4(const T & a,const T & b, const T& c, const T &d, T &r)     {
    r.copy_meta(a);    
    array_r_norm4df(a.DP(),b.DP(),c.DP(),d.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_rnorm4(const T & a,const T & b, const T& c, const T &d, T &r)     {
    r.copy_meta(a);    
    array_r_rnorm4df(a.DP(),b.DP(),c.DP(),d.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_normcdf(const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_normcdff(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_normcdinvf(const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_normcdinvf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_norm(int dim, const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_normf(dim,a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_rnorm(int dim, const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_rnormf(dim,a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_tgamma(const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_tgammaf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_y0(const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_y0f(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_y1(const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_y1f(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_yn(int n, const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_ynf(n,a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_copysign(const T & a,const T & b, T &r)     {
    r.copy_meta(a);    
    array_r_copysignf(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_erfc(const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_erfcf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_erfcinv(const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_erfcinvf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;    
}
template<typename T>
T& reg_erfcx(const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_erfcxf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_erf(const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_erff(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_erfinv(const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_erfinvf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}



template<typename T>
T& reg_fdim(const T & a,const T & b, T &r)     {
    r.copy_meta(a);    
    array_r_fdimf(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_fdim(const T& a,const float b, T& r)     {    
    r.copy_meta(a);    
    array_r_fdimf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_fdim(const T& a,const Scalar& b, T& r)     {    
    r.copy_meta(a);    
    array_r_fdimf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_fdim(const T & a,const GPUConst & b, T &r)     {
    r.copy_meta(a);    
    array_r_fdimf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_fdim(const T & a,const GPUScalar & b, T &r)     {
    r.copy_meta(a);    
    array_r_fdimf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}

template<typename T>
T& reg_nearbyint(const T & a, T &r)     {
    r.copy_meta(a);    
    array_r_nearbyintf(a.DP(),r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}


template<typename T>
T& reg_remainder(const T & a,const T & b, T &r)     {
    r.copy_meta(a);    
    array_r_remainderf(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_remainder(const T& a,const float b, T& r)     {    
    r.copy_meta(a);    
    array_r_remainderf_const(a.DP(),b,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T reg_remainder(const T& a,const Scalar& b, T& r)     {    
    r.copy_meta(a);    
    array_r_remainderf_const(a.DP(),b.value,r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_remainder(const T & a,const GPUConst & b, T &r)     {
    r.copy_meta(a);    
    array_r_remainderf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}
template<typename T>
T& reg_remainder(const T & a,const GPUScalar & b, T &r)     {
    r.copy_meta(a);    
    array_r_remainderf_scalar(a.DP(),b.DP(), r.DP(),a.Dim1(),a.Dim2(),a.Dim3(),a.Dim4());      
    return r;
}




///////////////////////////////////////////////////////////////////////////////////
// cublas
///////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////
// Blas Level1
//////////////////////////////////////////////////
// y = alpha*x[k] + y[j]
void axpy(Scalar& alpha, const GPUVector & a, GPUVector & b, int incx=1, int incy=1)    {    
    cublasStatus_t     status; 
    status = cublasSaxpy(cublas->handle,a.size(),&alpha.value,a.devPtr, incx, b.devPtr, incy);
    ASSERTOK(status);    
}
// y = alpha*x[k] + y[j]
void axpy(float _alpha, const GPUVector & a, GPUVector & b, int incx=1, int incy=1)    {    
    Scalar alpha(_alpha);
    cublasStatus_t     status; 
    status = cublasSaxpy(cublas->handle,a.size(),&alpha.value,a.devPtr, incx, b.devPtr, incy);
    ASSERTOK(status);    
}
// first index of greatest a[i]
int  amax(const GPUVector & v, int incx=1)    {
    int result=-1;
    cublasStatus_t     status; 
    status = cublasIsamax(cublas->handle,v.size(),v.devPtr, incx, &result);
    ASSERTOK(status);
    return result;
}
// first index of least a[i]
int amin(const GPUVector & v, int incx=1) {
    int result=-1;
    cublasStatus_t     status; 
    status = cublasIsamin(cublas->handle,v.size(),v.devPtr, incx, &result);
    ASSERTOK(status);    
    return result;
}
// x = sum(v)
Scalar  asum(const GPUVector & v, int incx=1) {
    Scalar result(0);
    cublasStatus_t     status;     
    status = cublasSasum(cublas->handle,v.size(),v.devPtr, incx, &result.value);
    ASSERTOK(status);
    return result;
}
// normalize v and return value
Scalar nrm2(const GPUVector & v, int incx=1) {        
    Scalar result(0);
    cublasStatus_t     status;     
    status = cublasSnrm2(cublas->handle,v.size(),v.devPtr, incx, &result.value);
    ASSERTOK(status);
    return result;
}

// dot x,y
Scalar dot(const GPUVector & x,GPUVector & y, int incx=1,int incy=1) {
    Scalar result(0);
    cublasStatus_t     status; 
    status = cublasSdot(cublas->handle,x.size(),x.devPtr, incx, y.devPtr,incy, &result.value);
    ASSERTOK(status);
    return result;
}
// rot x,y and return new vector
void rot(const GPUVector & x, GPUVector & y, Scalar& cosine, Scalar& sinus, int incx=1,int incy=1) {    
    cublasStatus_t     status; 
    status = cublasSrot(cublas->handle,x.size(),x.devPtr, incx, y.devPtr, incy, &cosine.value, &sinus.value);
    ASSERTOK(status);    
}
// make rotation matrix
void rotg(Scalar &a, Scalar &b, Scalar &cosine, Scalar &sinus) {                
    cublasStatus_t     status; 
    status = cublasSrotg(cublas->handle, &a.value, &b.value, &cosine.value, &sinus.value);
    ASSERTOK(status);
}
// rotate x,y and return new vector
void rotm(const GPUVector & x, GPUVector & y, Scalar&  param, int incx=1,int incy=1) {    
    cublasStatus_t     status; 
    status = cublasSrotm(cublas->handle,x.size(),x.devPtr, incx, y.devPtr, incy, &param.value);
    ASSERTOK(status);    
}
// create rotation matrix in parameters passed
void rotmg(Scalar &d1, Scalar &d2, Scalar &x1, Scalar &y1, Scalar &param) {
    cublasStatus_t     status; 
    status = cublasSrotmg(cublas->handle,&d1.value,&d2.value,&x1.value,&y1.value,&param.value);
    ASSERTOK(status);
}
// scal v with alpha  and return new vector
void scal(GPUVector & v, Scalar& alpha,int incx=1) {    
    cublasStatus_t     status; 
    status = cublasSscal(cublas->handle, v.size(), &alpha.value, v.devPtr, incx);
    ASSERTOK(status);    
}
// scal v with alpha  and return new vector
void scal(GPUVector & v, float alpha,int incx=1) {    
    cublasStatus_t     status; 
    status = cublasSscal(cublas->handle, v.size(), &alpha, v.devPtr, incx);
    ASSERTOK(status);    
}

// swap this with v and return new vector
void swap(GPUVector & src, GPUVector & dst, int incx=1, int incy=1) {    
    cublasStatus_t     status; 
    status = cublasSswap(cublas->handle, src.N, src.devPtr, incx, dst.devPtr, incy);    
}
// copy src to dst
void copy(GPUVector & src, GPUVector & dst, int incx=1,int incy=1)    {    
    cublasStatus_t     status; 
    status = cublasScopy(cublas->handle,src.size(),src.devPtr, incx, dst.devPtr, incy);
    ASSERTOK(status);    
}


/////////////////////////////////////
// Blas 2 and 3 
/////////////////////////////////////

// y = alpha * op(A)*x + beta * y
// op(A) = A if cublas_op_n 
// op(A) = A transpose if cublas_op_t 
// op(A) = A^H if cublas_op_h
void gbmv(cublasOperation_t trans, int m, int n, float alpha, const GPUMatrix &A, int lda, int kl, int ku, const GPUVector &x, float beta, GPUVector& y, int incx=1,int incy=1)
{         
    cublasStatus_t status = cublasSgbmv(cublas->handle,trans,m,n,kl,ku,&alpha,A.devPtr,lda,x.devPtr,incx, &beta, y.devPtr,incy);
    ASSERTOK(status);    
}

// r = alpha * op(A) * x + beta * y 
void gemv(cublasOperation_t trans, int m, int n, float alpha, const GPUMatrix &A, int lda, const GPUVector &x, float beta, GPUVector &y, int incx=1,int incy=1)
{            
    cublasStatus_t status = cublasSgemv(cublas->handle,trans,m,n,&alpha,A.devPtr,lda,x.devPtr,incx,&beta,y.devPtr,incy);
    ASSERTOK(status);    
}


// y = alpha * x * transpose(y) if  ger,geru
// y = alpha * x * H(y) if gerc
void ger(int m, int n, float alpha, GPUVector & x, GPUVector &y, GPUMatrix & A, int incx=1,int incy=1, int lda=-1)
{    
    if(lda == -1) lda = A.M;        
    cublasStatus_t status = cublasSger(cublas->handle,m,n,&alpha,x.devPtr,incx,y.devPtr,incy,A.devPtr,lda);
    ASSERTOK(status);    
}
// y = alpha * A * x + beta * y 
void sbmv(cublasFillMode_t uplo, int n, int k, float alpha, GPUMatrix & A, int lda,GPUVector &x, float beta, GPUVector & y, int incx=1, int incy=1)
{        
    cublasStatus_t status = cublasSsbmv(cublas->handle,uplo,n,k,&alpha,A.devPtr,lda,x.devPtr,incx,&beta,y.devPtr,incy);
    ASSERTOK(status);    
}

// A = alpha*x*tranpose(x) + A 
void spr(cublasFillMode_t uplo, int n, const float alpha, GPUVector & v, GPUMatrix & AP, int incx=1)
{    
    cublasStatus_t status = cublasSspr(cublas->handle,uplo,n,&alpha,v.devPtr,incx, AP.devPtr);
    ASSERTOK(status);    
}

// A = alpha*(x*tranpose(y) + y*transpose(x)) + A
void spr2(cublasFillMode_t uplo, int n, const float alpha, GPUVector & x, GPUVector &y, GPUMatrix & AP, int incx=1, int incy=1)
{    
    cublasStatus_t status = cublasSspr2(cublas->handle,uplo,n,&alpha,x.devPtr,incx,y.devPtr,incy,AP.devPtr);
    ASSERTOK(status);    
}

// y = alpha*A*x + beta*y 
void symv(cublasFillMode_t uplo, int n, float alpha, GPUMatrix & A, int lda, GPUVector &x, float beta, GPUVector &y, int incx=1,int incy=1)
{    
    cublasStatus_t status = cublasSsymv(cublas->handle,uplo,n,&alpha,A.devPtr,lda,x.devPtr,incx,&beta,y.devPtr,incy);
    ASSERTOK(status);    
}

// A = alpha*x*tranpose(x) + A
void syr(cublasFillMode_t uplo, int n, float alpha, GPUVector &x, GPUMatrix &A, int lda, int incx=1)
{        
    cublasStatus_t status = cublasSsyr(cublas->handle, uplo, n, &alpha, x.devPtr, incx, A.devPtr, lda);
    ASSERTOK(status);    
}

// A = alpha*(x*transpose(y) + y*transpose(x)) + A
void syr2(cublasFillMode_t uplo, float alpha, GPUVector & x, GPUVector & y, GPUMatrix & A, int lda, int incx=1,int incy=1)
{    
    cublasStatus_t status = cublasSsyr2(cublas->handle,uplo,A.M*A.N, &alpha,  x.devPtr,incx, y.devPtr,incy, y.devPtr,lda );
    ASSERTOK(status);    
}

// op(A)*x = b
void tbmv(cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, GPUMatrix & A, int lda, GPUVector &x, int incx=1)
{    
    cublasStatus_t status = cublasStbmv(cublas->handle, uplo, trans, diag,n,k,A.devPtr,lda, x.devPtr,incx);
    ASSERTOK(status);    
}

// b = op(A)x
void tbsv(cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int k, GPUMatrix & A, int lda, GPUVector & x, int incx=1)
{    
    cublasStatus_t status = cublasStbsv(cublas->handle,uplo,trans,diag,A.M*A.N,k,A.devPtr,lda,x.devPtr,incx);
    ASSERTOK(status);    
}

// x = op(A)x
void trmv(cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, GPUMatrix & A, int lda, GPUVector & x, int incx=1)
{    
    cublasStatus_t status = cublasStrmv(cublas->handle,uplo,trans,diag,n,A.devPtr,lda,x.devPtr,incx);
    ASSERTOK(status);    
}

// op(A)*x = b
void trsv(cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, GPUMatrix & A, int lda, GPUVector & x, int incx=1)
{        
    cublasStatus_t status = cublasStrsv(cublas->handle,uplo,trans,diag,n,A.devPtr,lda,x.devPtr,incx);
    ASSERTOK(status);    
}

// general matrix multiply C=alpha*AB + beta*C
void gemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, const GPUMatrix &A, int lda,  const GPUMatrix &B, int ldb, float beta,  GPUMatrix &C, int ldc)
{       
    cublasStatus_t status = cublasSgemm(cublas->handle,transa,transb,m,n,k,&alpha,A.devPtr,lda,B.devPtr,ldb,&beta,C.devPtr,ldc);
    ASSERTOK(status);    
}

// batched gemm multiply
void gemmBatched(cublasOperation_t transa, cublasOperation_t transb, float alpha, int m, int n, int k, float** A, int lda, float **B, int ldb, float beta, float **C, int ldc, int batchCount) {
    cublasStatus_t status = cublasSgemmBatched(cublas->handle, transa, transb, m,n,k, &alpha, A,lda, B,ldb,&beta, C,ldc,batchCount);
    ASSERTOK(status);
}
void gemmStridedBatched(cublasOperation_t transa, cublasOperation_t transb, float alpha, int m, int n, int k, float* A, int lda, int strideA, float *B, int ldb, int strideB, float beta, float *C, int ldc, int strideC, int batchCount) {
    cublasStatus_t status = cublasSgemmStridedBatched(cublas->handle, transa, transb,m,n,k, &alpha, A,lda,strideA,B,ldb,strideB,&beta,C,ldc,strideC,batchCount);
    ASSERTOK(status);
}

// symetric matrix multiplication
// if side == left then C = alpha*A*B + beta*C 
// else is side == right then C = alpha*B*C + beta*C
void symm(cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, float alpha, GPUMatrix &A, int lda, GPUMatrix &B, int ldb, float beta, GPUMatrix &C, int ldc)
{    
    cublasStatus_t status = cublasSsymm(cublas->handle, side, uplo, m,n, &alpha, A.devPtr,lda,B.devPtr,ldb,&beta,C.devPtr,ldc);
    ASSERTOK(status);
}

// C = alpha * op(A) * op_transpose(A) + beta * C
// symetric rank update
void syrk(cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, GPUMatrix &A, int lda, float beta, GPUMatrix &C, int ldc)
{
    if(lda == -1) lda = A.M;
    if(ldc == -1) ldc = C.M;
    cublasStatus_t status = cublasSsyrk(cublas->handle,uplo,trans,n,k,&alpha,A.devPtr,lda,&beta,C.devPtr,ldc);
    ASSERTOK(status);
}

// C = alpha * (op(A) * op_transpose(B)) + op(B)+op_transposed(A) + beta*C 
void syr2k(cublasFillMode_t uplo, cublasOperation_t trans, float alpha, int n, int k, GPUMatrix & A, int lda, GPUMatrix & B, int ldb, float beta, GPUMatrix & C, int ldc)
{
    cublasStatus_t status = cublasSsyr2k(cublas->handle,uplo,trans,n,k,&alpha, A.devPtr,lda,B.devPtr,ldb,&beta,C.devPtr,ldc);
    ASSERTOK(status);
}
// C = alpha * op(A) * op_transposed(B) + beta * C
// symetric rank update variation
void syrkx(cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, GPUMatrix & A, int lda, GPUMatrix & B, int ldb, float beta, GPUMatrix &C, int ldc)
{
    cublasStatus_t status = cublasSsyrkx(cublas->handle,uplo,trans,n,k,&alpha,A.devPtr,lda,B.devPtr,ldb,&beta,C.devPtr,ldc);
    ASSERTOK(status);
}

// if side == left C = alpha * op(A)*B 
// else C = alpha*B*op(A)
void trmm(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, float alpha, GPUMatrix & A, int lda, GPUMatrix & B, int ldb, GPUMatrix & C, int ldc)
{
    cublasStatus_t status = cublasStrmm(cublas->handle,side,uplo,trans,diag,m,n,&alpha,A.devPtr,lda,B.devPtr,ldb,C.devPtr,ldc);
    ASSERTOK(status);
}

// if side == left op(A)X = alpha * B
// else X * op(A) = alpha * B
void trsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n,float alpha, GPUMatrix & A, int lda, GPUMatrix & B, int ldb)
{
    cublasStatus_t status = cublasStrsm(cublas->handle,side,uplo,trans,diag,m,n,&alpha,A.devPtr,lda,B.devPtr,ldb);
    ASSERTOK(status);
}

void trsmBatched(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,int m,int n, float alpha, float** A, float** B, int batchCount,int lda,int ldb)
{    
    cublasStatus_t status = cublasStrsmBatched(cublas->handle,side,uplo,trans,diag,m,n,&alpha,A,lda,B,ldb,batchCount);
    ASSERTOK(status);
}

// C = alpha * op(A) + beta * op(B)
void geam(cublasOperation_t transa, cublasOperation_t transb, int m, int n, float alpha, const GPUMatrix & A, int lda, const GPUMatrix & B, int ldb, float beta, GPUMatrix & C, int ldc)
{
    cublasStatus_t status = cublasSgeam(cublas->handle,transa,transb,m,n,&alpha,A.devPtr,lda,&beta,B.devPtr,ldb,C.devPtr,ldc);
    ASSERTOK(status);
}

void dgmm(cublasSideMode_t mode, int m, int n, GPUMatrix & A, int lda, GPUVector &x, GPUMatrix & C, int ldc, int incx=1)
{    
    cublasStatus_t status = cublasSdgmm(cublas->handle,mode,m,n,A.devPtr,lda, x.devPtr,incx, C.devPtr, ldc);
    ASSERTOK(status);
}

void gemmEx(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, void* A, cudaDataType_t Atype, int lda, const void * B, cudaDataType_t Btype, int ldb, float beta, void * C, cudaDataType_t Ctype, int ldc)
{
    cublasStatus_t status = cublasSgemmEx(cublas->handle,transa,transb,m,n,k,&alpha,A,Atype,lda,B,Btype,ldb,&beta,C,Ctype,ldc);
    ASSERTOK(status);
}



///////////////////////////////////////////////
// GPUVector
///////////////////////////////////////////////


GPUVector GPUVector::operator + (GPUVector & y)    {
    GPUVector A(*this), B(y);    
    axpy(1.0f,A,B);
    return B;
}
GPUVector GPUVector::operator - (GPUVector & y)    {
    GPUVector A(*this), B(y);    
    axpy(-1.0f,B,A);
    return A;
}
GPUVector GPUVector::operator * (GPUVector & y)    {
    GPUVector r(*this);
    cublasStatus_t     status; 
    // hadamard product
    status = cublasSdgmm(cublas->handle,CUBLAS_SIDE_RIGHT,1, size(), devPtr, 1, y.devPtr, 1, r.devPtr,1);
    return r;
}
GPUVector GPUVector::operator / (GPUVector & y)    {    
    GPUVector       r(*this / y);
    return r;
}
GPUVector GPUVector::operator % (GPUVector & y)    {    
    GPUVector       r(*this % y);    
    return r;
}

GPUVector GPUVector::operator + (float v)    {    
    GPUVector A(*this), B(size());
    B.fill(v);    
    axpy(1.0f,A,B);
    return B;
}
GPUVector GPUVector::operator - (float v)    {
    GPUVector A(*this), B(size());
    B.fill(v);    
    axpy(-1.0f,B,A);
    return A;
}
GPUVector GPUVector::operator * (float v)    {                
    GPUVector A(*this);
    scal(A,v);        
    return A;
}
GPUVector GPUVector::operator / (float v)    {
    Scalar rec(v);
    GPUVector A(*this);
    scal(A,rec);        
    return A;
}
GPUVector GPUVector::operator % (float v)    {
    Scalar rec(v);
    GPUVector A(*this);
    GPUVector B(size());
    B.fill(v);
    GPUVector C(size());
    C = A % B;
    return C;
}
float GPUVector::dot(const GPUVector & b){
    GPUVector A(*this);
    GPUVector B(b);    
    Scalar s = ::dot(A,B);
    return s.value;
}
float GPUVector::mag() {
    return nrm2(*this).value;
}


GPUVector GPUVector::operator + (const GPUConst & v) {
    float *p = vector_addf_scalar(devPtr,v.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator - (const GPUConst & v){
    float *p = vector_subf_scalar(devPtr,v.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator * (const GPUConst & v){
    float *p = vector_mulf_scalar(devPtr,v.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator / (const GPUConst & v){
    float *p = vector_divf_scalar(devPtr,v.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator % (const GPUConst & v){
    float *p = vector_modf_scalar(devPtr,v.devPtr,size());
    return GPUVector(p,size());
}

GPUVector GPUVector::operator + (const GPUScalar & s){
    float *p = vector_addf_scalar(devPtr,s.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator - (const GPUScalar & s){
    float *p = vector_subf_scalar(devPtr,s.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator * (const GPUScalar & s){
    float *p = vector_mulf_scalar(devPtr,s.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator / (const GPUScalar & s){
    float *p = vector_divf_scalar(devPtr,s.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator % (const GPUScalar & s){
    float *p = vector_modf_scalar(devPtr,s.devPtr,size());
    return GPUVector(p,size());
}

///////////////////////////////////////////////
// GPUMatrix
///////////////////////////////////////////////


void reg_matmul(const GPUMatrix &a, const GPUMatrix & b,GPUMatrix & C, bool transa=false,bool transb=false, float alpha=1.0,float beta=0.0) {         
    if(C.Dim1() != a.Dim1() || C.Dim2() != b.Dim2())
        C.resize(a.Dim1(),b.Dim2());
    cublasOperation_t ta = transa? CUBLAS_OP_T:CUBLAS_OP_N;
    cublasOperation_t tb = transb? CUBLAS_OP_T:CUBLAS_OP_N;    
    int m = a.Dim1();
    int n = a.Dim2();
    int k = b.Dim2();
    gemm(ta,tb,n,m,k,alpha,b,n,a,k,beta,C,n);                                
}

GPUVector GPUMatrix::matvec(const GPUVector & v,bool transa, float alpha, float beta ) {    
    GPUVector R(v.size());
    int m = Dim1();
    int n = Dim2();
    cublasOperation_t ta = transa? CUBLAS_OP_N:CUBLAS_OP_T;
    gemv(ta,n,m,alpha,*this,n,v,beta,R);
    return R;
}

// C^T = B^T * A^T
// assumes a and b are in row major order
GPUMatrix matmul(const GPUMatrix & a, const GPUMatrix & b, float alpha=1.0, float beta=0.0) {         
    GPUMatrix C(a.M,b.N);
    cublasOperation_t ta = CUBLAS_OP_N;
    cublasOperation_t tb = CUBLAS_OP_N;      
    int m = a.Dim1();
    int k = a.Dim2();
    int n = b.Dim2();    
    gemm(ta,tb,n,m,k,alpha,b,n,a,k,beta,C,n);                        
    return C;    
}

// because it is in row major it uses identity C^t = B^t * A^t 
// the transpose flags do not work.
GPUMatrix GPUMatrix::matmul(const GPUMatrix & b,bool transa,bool transb,bool transc, float alpha,float beta) {         
    GPUMatrix C(M,b.N);
    cublasOperation_t ta = transa? CUBLAS_OP_T:CUBLAS_OP_N;
    cublasOperation_t tb = transb? CUBLAS_OP_T:CUBLAS_OP_N;    
    int m = Dim1();
    int n = Dim2();
    int k = b.Dim2();
    gemm(ta,tb,n,m,k,alpha,b,n,*this,k,beta,C,n);                            
    return C;    
}


// C^T = A*B
// A and B are in column major order
GPUMatrix matmulTT(const GPUMatrix & a,const GPUMatrix & b, float alpha=1.0, float beta=0.0) {         
    assert(a.N == b.M);    
    cublasOperation_t ta = CUBLAS_OP_T;
    cublasOperation_t tb = CUBLAS_OP_T;      
    GPUMatrix C(a.N,b.M);            
    int m = a.Dim1();
    int k = a.Dim2();
    int n = b.Dim2(); 
    gemm(ta,tb,n,m,k,alpha,b,n,a,k,beta,C,n);                                            
    return C;    
}

// C^T = B^T*A
// A is in column major order
GPUMatrix matmulTN(const GPUMatrix & a,const GPUMatrix & b, float alpha=1.0, float beta=0.0) {         
    assert(a.N == b.N);
    GPUMatrix A(a),B(b);
    cublasOperation_t ta = CUBLAS_OP_N;
    cublasOperation_t tb = CUBLAS_OP_T;               
    GPUMatrix C(A.N,B.N);        
    int m = A.Dim1();
    int k = A.Dim2();
    int n = B.Dim2(); 
    gemm(ta,tb,n,k,m,alpha,B,n,A,k,beta,C,n); 
    return C;        
}
// C^T = B*A^T
// A is in column major order
GPUMatrix matmulNT(const GPUMatrix & a,const GPUMatrix & b, float alpha=1.0, float beta=0.0) {         
    assert(a.M == b.M);    
    cublasOperation_t ta = CUBLAS_OP_T;
    cublasOperation_t tb = CUBLAS_OP_N;            
    GPUMatrix C(a.M,b.M);        
    int m = a.Dim1();
    int k = a.Dim2();
    int n = b.Dim2(); 
    gemm(ta,tb,m,n,k,alpha,b,n,a,k,beta,C,n);                        
    return C;    
}

GPUMatrix matmul_cuda(const GPUMatrix & a,const GPUMatrix & b)
{
    float * p = matrix_multiplyf(a.devPtr, b.devPtr, a.Dim1(),a.Dim2(),b.Dim2());
    return GPUMatrix(p,a.Dim1(),b.Dim2());
}

GPUVector matvec(const GPUMatrix & a, const GPUVector & b,bool transa=false, float alpha=1.0, float beta=0.0) {    
    GPUVector R(b.size());
    int m = a.Dim1();
    int n = a.Dim2();
    cublasOperation_t ta = transa? CUBLAS_OP_T:CUBLAS_OP_N;    
    gemv(CUBLAS_OP_N,m,n,alpha,a,m,b,beta,R);
    return R;
}

GPUMatrix geam_transpose(const GPUMatrix & x) {    
    GPUMatrix r(x);        
    r.swap_order();
    int m = x.Dim1();
    int n = x.Dim2();    
    geam(CUBLAS_OP_T,CUBLAS_OP_N,m,n,1.0,x,n,x,m,0.0,r,m);    
    return r;    
}
GPUMatrix transpose_cuda(const GPUMatrix & x) {    
    float *p = matrix_transposef(x.devPtr,x.Dim1(),x.Dim2());
    return GPUMatrix(p,x.Dim2(),x.Dim1());
}
GPUMatrix GPUMatrix::t() {    
    GPUMatrix r(*this);        
    r.swap_order();
    int m = Dim1();
    int n = Dim2();    
    geam(CUBLAS_OP_T,CUBLAS_OP_N,m,n,1.0,*this,n,*this,m,0.0,r,m);    
    return r;    
}

GPUMatrix transpose(GPUMatrix & a){
    return a.t();
}

GPUMatrix GPUMatrix::operator + (const GPUMatrix & b) {         
    assert(M == b.M && N == b.N);
    GPUMatrix r(Dim1(),Dim2());
    int m = Dim1();
    int k = Dim2();    
    int n = Dim2();
    geam(CUBLAS_OP_N,CUBLAS_OP_N,n,m,1.0,b,n,*this,k,1.0,r,n);
    return r;
}

GPUMatrix GPUMatrix::operator - (const GPUMatrix & b) { 
    assert(M == b.M && N == b.N);
    GPUMatrix r(Dim1(),Dim2());
    int m = Dim1();
    int k = Dim2();    
    int n = Dim2();
    geam(CUBLAS_OP_N,CUBLAS_OP_N,n,m,-1.0,b,n,*this,k,1.0,r,n);
    return r;
}

GPUMatrix GPUMatrix::operator * (const GPUMatrix & m) 
{         
    return ::matmul(*this,m);
}

GPUMatrix hadamard(const GPUMatrix & a, const GPUMatrix & b) {
    assert(a.M == b.M && a.N == b.N);
    float * p = matrix_hadamardf(a.devPtr,b.devPtr,a.Dim1(),a.Dim2(),b.Dim2());
    return GPUMatrix(p,a.Dim1(),a.Dim2());
}


void reg_hadamard(const GPUMatrix & a, const GPUMatrix & b, GPUMatrix & c) {
    assert(a.M == b.M && a.N == b.N);
    c.resize(a.Dim1(),a.Dim2());
    matrix_r_hadamardf(a.devPtr,b.devPtr,c.devPtr,a.Dim1(),a.Dim2(),b.Dim2());    
}


GPUMatrix GPUMatrix::operator + (const GPUConst & v) {
    float *p = vector_addf_scalar(devPtr,v.devPtr,size());
    return GPUMatrix(p,Dim1(),Dim2());
}
GPUMatrix GPUMatrix::operator - (const GPUConst & v) {
    float *p = vector_subf_scalar(devPtr,v.devPtr,size());
    return GPUMatrix(p,Dim1(),Dim2());
}
GPUMatrix GPUMatrix::operator * (const GPUConst & v) {
    float *p = vector_mulf_scalar(devPtr,v.devPtr,size());
    return GPUMatrix(p,Dim1(),Dim2());
}
GPUMatrix GPUMatrix::operator / (const GPUConst & v) {
    float *p = vector_divf_scalar(devPtr,v.devPtr,size());
    return GPUMatrix(p,Dim1(),Dim2());
}
GPUMatrix GPUMatrix::operator % (const GPUConst & v) {
    float *p = vector_modf_scalar(devPtr,v.devPtr,size());
    return GPUMatrix(p,Dim1(),Dim2());
}
GPUMatrix GPUMatrix::operator + (const GPUScalar & s) {
    float *p = vector_addf_scalar(devPtr,s.devPtr,size());
    return GPUMatrix(p,Dim1(),Dim2());
}
GPUMatrix GPUMatrix::operator - (const GPUScalar & s) {
    float *p = vector_subf_scalar(devPtr,s.devPtr,size());
    return GPUMatrix(p,Dim1(),Dim2());
}
GPUMatrix GPUMatrix::operator * (const GPUScalar & s) {
    float *p = vector_mulf_scalar(devPtr,s.devPtr,size());
    return GPUMatrix(p,Dim1(),Dim2());
}
GPUMatrix GPUMatrix::operator / (const GPUScalar & s) {
    float *p = vector_divf_scalar(devPtr,s.devPtr,size());
    return GPUMatrix(p,Dim1(),Dim2());
}
GPUMatrix GPUMatrix::operator % (const GPUScalar & s) {
    float *p = vector_modf_scalar(devPtr,s.devPtr,size());
    return GPUMatrix(p,Dim1(),Dim2());
}



int current_stream = 0;
CudaStream cuda_streams[16];

cudaStream_t get_cuda_stream() {
    return cuda_streams[current_stream].stream;
}

void set_stream(int streamid) {
    assert(streamid >= 0 && streamid < 16);
    current_stream = streamid;    
    cublas->SetStream(cuda_streams[current_stream]);    
}
int get_stream() { 
    return current_stream; 
}

cudaStream_t random_stream() {
    return cuda_streams[rand() % 16].stream;
}

#endif
