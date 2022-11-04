///////////////////////////////////////////////////////////////////////////////////////////////
// viperfish (cublas++)
// Vector and Matrix should derive from array
// It would reduce the number of functions down.
///////////////////////////////////////////////////////////////////////////////////////////////
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
#include "vector_float.h"

struct GPUArray;
struct GPUArrayView;
struct GPUVector;
struct GPUMatrix;
struct GPUMatrixView;


#define _MV(x) (x).vector->M
#define _MM(x) (x).matrix->M
#define _NM(x) (x).matrix->N


// this can be called from interface
void    clear_cache();

// 1 2 3 
// 4 5 6 
// 7 8 9 

// 1 4 7 
// 2 5 8
// 3 6 9

// i dont want it in column because it makes loading it awkward from C++.
#define IDX2(i,j,n) ((j*n)+i)
#define IDX3(i,j,k,n,o) ((k*o)+(j*n)+i)
#define IDX4(i,j,k,w,n,o,p) ((w*p)+(k*o)+(j*n)+i)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Cell
// It serves no actual purpose in the program. 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Cell
{
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    uint8_t ui8;
    uint16_t ui16;
    uint32_t ui32;
    uint64_t ui64;
    float    f;
    double   d;
    std::string str;
    void     *ptr;        

    Cell() {
        i8=0;
        i16=0;
        i32=0;
        i64=0;
        ui8=0;
        ui16=0;
        ui32=0;
        ui64=0;
        f = 0;
        d = 0;
        str="";
        ptr = NULL;
    }
};

#define ASSERTOK(status) (assert(status == CUBLAS_STATUS_SUCCESS))


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuda/cublas
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct CudaStream
{
    cudaStream_t stream;
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

template<typename T>
struct GPUConst 
{
    std::shared_ptr<T>       dev_ptr;    
    T*                       devPtr;

    GPUConst(float value) {
        T * temp;
        cudaError_t err = cudaMalloc((void**)&temp,sizeof(T));
        if(err != cudaSuccess) {
            std::cout << cudaGetErrorString(err) << std::endl;
        }
        assert(err == cudaSuccess);          
        dev_ptr  = { temp, cudaFree};
        devPtr = dev_ptr.get();        
    }   
    ~GPUConst() {
        return_memory(sizeof(T), devPtr);
    }
};

template<typename T>
struct GPUScalar
{
    std::shared_ptr<T>       dev_ptr;    
    T*                       devPtr;
    
    GPUScalar(T value) {
        T * temp;
        cudaError_t err = cudaMalloc((void**)&temp,sizeof(T));
        if(err != cudaSuccess) {
            std::cout << cudaGetErrorString(err) << std::endl;
        }
        assert(err == cudaSuccess);          
        dev_ptr  = { temp, cudaFree};
        devPtr = dev_ptr.get();        
    }
    GPUScalar(T * p_dev) {
        devPtr = p_dev;
        dev_ptr = { p_dev, cudaFree};
    }
    ~GPUScalar() {
        return_memory(sizeof(T));
    }

    GPUScalar& operator = (const GPUScalar &s) {
        dev_ptr.reset();
        devPtr = s.devPtr;
        dev_ptr = s.dev_ptr;
        return *this;
    }
    void set_value(const float v) {
        cudaMemcpy(devPtr,&v,sizeof(float),cudaMemcpyHostToDevice);
    }
    float get_value() {
        float r;
        cudaMemcpy(&r,devPtr,sizeof(float),cudaMemcpyDeviceToHost);
        return r;
    }    

    GPUScalar operator + (const GPUScalar & s) {
        float * p = vector_add(devPtr,s.devPtr,1);
        return GPUScalar(p);
    }
    GPUScalar operator - (const GPUScalar & s) {
        float * p = vector_sub(devPtr,s.devPtr,1);
        return GPUScalar(p);
    }
    GPUScalar operator * (const GPUScalar & s) {
        float * p = vector_mul(devPtr,s.devPtr,1);
        return GPUScalar(p);
    }
    GPUScalar operator / (const GPUScalar & s) {
        float * p = vector_div(devPtr,s.devPtr,1);
        return GPUScalar(p);
    }
    GPUScalar operator % (const GPUScalar & s) {
        float * p = vector_mod(devPtr,s.devPtr,1);
        return GPUScalar(p);
    }
    
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPUArray (float)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class GPUArray 
{
private:

    void init() {       
        float * temp = find_memory(size()*sizeof(T));
        if(temp == NULL) {
            cudaError_t err = cudaMalloc((void**)&temp,M*N*O*P*sizeof(T));                              
            if(err != cudaSuccess) {
                std::cout << "M=" << M << std::endl;
                std::cout << "N=" << N << std::endl;
                std::cout << "O=" << O << std::endl;
                std::cout << "P=" << P << std::endl;
                std::cout << cudaGetErrorString(err) << std::endl;
            }
            assert(err == cudaSuccess);          
        }
        T * phost = (T*)calloc(M*N*O*P,sizeof(T));
        assert(phost != NULL);            
        host_ptr = { phost, free };
        devPtr = temp;
        host   = host_ptr.get();
    }
    void rand(float min, float max) {
        typedef std::chrono::high_resolution_clock myclock;
        myclock::time_point beginning = myclock::now();
        myclock::duration d = myclock::now() - beginning;
        unsigned seed = d.count();
        std::default_random_engine generator(seed);
        std::uniform_real_distribution<double> distribution(min,max);                
        for(size_t i = 0; i < M*N*O*P; i++) host[i] = distribution(generator);
        cudaMemcpy(devPtr,host,M*N*O*P*sizeof(T),cudaMemcpyHostToDevice);    
    }

public:
    
    std::shared_ptr<T>       host_ptr;
    T*                       devPtr;
    T*                       host;
    size_t  M;
    size_t  N;
    size_t  O;
    size_t  P;

    GPUArray() {
        devPtr = nullptr;
        host   = nullptr;
    }
    GPUArray(size_t m)    {
        M = m;
        N = 1;        
        O = 1;
        P = 1;
        init();
    }
    GPUArray(size_t m, size_t n)    {
        M = m;
        N = n;        
        O = 1;
        P = 1;
        init();
    }
    GPUArray(size_t m, size_t n, size_t o)    {
        M = m;
        N = n;        
        O = o;
        P = 1;
        init();
    }
    GPUArray(size_t m, size_t n, size_t o, size_t p)    {
        M = m;
        N = n;        
        O = o;
        P = p;
        init();
    }
    GPUArray(const std::vector<T> & data, size_t m, size_t n=1, size_t o=1, size_t p=1)    {
        M = m;
        N = n;        
        O = o;
        P = p;
        init();
        cudaMemcpy(devPtr,data.data(),size()*sizeof(T),cudaMemcpyHostToDevice);
    }
    // p is on device
    GPUArray(T * devptr, size_t m, size_t n=1,size_t o=1,size_t p=1)   {
        M = m;
        N = n;
        O = o;
        P = p;
        host = (T*)calloc(M*N*O*P,sizeof(T));        
        host_ptr = { host, free };
        devPtr = devptr;
        host   = host_ptr.get();
    }
    GPUArray(const GPUArray & a) {
        M = a.M;
        N = a.N;
        O = a.O;
        P = a.P;        
        // it should not copy it here.
        // devPtr = a.devPtr;
        // host   = a.host;
        // dev_ptr = a.dev_ptr;
        // host_ptr = a.host_ptr;
        init();
        cudaMemcpy(devPtr,a.devPtr,size()*sizeof(T),cudaMemcpyDeviceToDevice);
    }

    ~GPUArray() {
        return_memory(size(),devPtr);        
    }

    GPUArray& operator = (const GPUArray & a)    {      
        // it creates a new image because the devPtr can't be shared at this time.
        M = a.M;
        N = a.N;
        O = a.O;
        P = a.P;            
        if(devPtr != nullptr) return_memory(size()*sizeof(T),devPtr);
        if(host != nullptr) host_ptr.reset();        
        init();
        cudaMemcpy(devPtr,a.devPtr,size()*sizeof(T),cudaMemcpyDeviceToDevice);        
        return *this;
    }
    
    GPUArray<float> operator -() {        
        float * p = vector_mul_const(devPtr,-1.0f,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray<float> operator + (const GPUArray & a)    {
        assert(size() == a.size());
        float * p = vector_add(devPtr,a.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray<float> operator - (const GPUArray & a)    {
        assert(size() == a.size());
        float * p = vector_sub(devPtr,a.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray<float> operator * (const GPUArray & a)    {
        assert(size() == a.size());
        float * p = vector_mul(devPtr,a.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray<float> operator / (const GPUArray & a)    {
        assert(size() == a.size());
        float * p = vector_div(devPtr,a.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;        
    }
    GPUArray<float> operator % (const GPUArray & a)    {
        assert(size() == a.size());
        float * p = vector_mod(devPtr,a.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }    
    
    GPUArray<float> operator + (const GPUConst<float> & c)    {
        float * p = vector_add_scalar(devPtr,c.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray<float> operator - (const GPUConst<float> & c)    {
        float * p = vector_sub_scalar(devPtr,c.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray<float> operator * (const GPUConst<float> & c)    {
        float * p = vector_mul_scalar(devPtr,c.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray operator / (const GPUConst & c)    {
        float * p = vector_div_scalar(devPtr,c.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray operator % (const GPUConst & c)    {
        float * p = vector_mod_scalar(devPtr,c.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }

    GPUArray operator + (const GPUScalar & s)    {
        float * p = vector_add_scalar(devPtr,s.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray operator - (const GPUScalar & s)    {
        float * p = vector_sub_scalar(devPtr,s.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray operator * (const GPUScalar & s)    {
        float * p = vector_mul_scalar(devPtr,s.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray operator / (const GPUScalar & s)    {
        float * p = vector_div_scalar(devPtr,s.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;        
    }
    GPUArray operator % (const GPUScalar & s)    {
        float * p = vector_mod_scalar(devPtr,s.devPtr,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }

    GPUArray operator + (float v)    {
        float * p = vector_add_const(devPtr,v,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray operator - (float v)    {        
        float * p = vector_sub_const(devPtr,v,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray operator * (float v)    {
        float * p = vector_mul_const(devPtr,v,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    GPUArray operator / (float v)    {
        float * p = vector_div_const(devPtr,v,size());
        GPUArray r(p,M,N,O,P);
        return r;        
    }
    GPUArray operator % (float v)    {
        float * p = vector_mod_const(devPtr,v,size());
        GPUArray r(p,M,N,O,P);
        return r;
    }
    
    void download_host()     {
        cudaMemcpy(host,devPtr,M*N*O*P*sizeof(float),cudaMemcpyDeviceToHost);
    }
    void upload_device()     {
        cudaMemcpy(devPtr,host,M*N*O*P*sizeof(float),cudaMemcpyHostToDevice);
    }
    size_t size() const { 
        return M*N*O*P; 
    }
    void zero()    {
        //for(size_t i = 0; i < M*N*O*P; i++) host[i] = 0;
        memset(host,0x0,M*N*O*P*sizeof(float));
        cudaMemcpy(devPtr,host,M*N*O*P*sizeof(float),cudaMemcpyHostToDevice);
    }
    void ones()    {
        float o = 1.0f;
        int one = *(int*)&o;
        //for(size_t i = 0; i < M*N*O*P; i++) host[i] = 1;
        memset(host,one,size()*sizeof(float));
        cudaMemcpy(devPtr,host,M*N*O*P*sizeof(float),cudaMemcpyHostToDevice);
    }
    
    void randu() {
        rand(0.0f,1.0f);
    }
    void random(float min, float max) {
        rand(min,max);
    }
    void fill(const float val)     {
        for(size_t i = 0; i < M*N*O*P; i++) host[i] = val;
        cudaMemcpy(devPtr,host,M*N*O*P*sizeof(float),cudaMemcpyHostToDevice);
    }


    bool has_nans() 
    {        
        bool nans = false;
        download_host();
        for(size_t i = 0; i < M*N; i++) 
        {
            if(isnan(host[i])) {
                std::cout << "NaN detected" << std::endl;
                nans = true;
            }
        }
        return nans;
    }
    // to use these you must download_host() first.
    // when you are done you must upload_device()
    // if you do not first download it will not be in sync with the device
    // if you do not upload it back to the device it will not be in sync with the device
    float& operator[](array_index pos)    {        
        assert(pos < M*N*O*P);
        if(pos < 0) pos += M*N*O*P;
        return host[pos];        
    }
    float& operator()(GPUIndex & idx)    {
        assert(idx.i < M);
        assert(idx.j < N);        
        assert(idx.k < O);        
        assert(idx.w < P);         
        array_index i = idx.i;
        array_index j = idx.j;
        array_index k = idx.k;
        array_index w = idx.w;                
        if(i < 0) i += M;        
        if(j < 0) j += N;
        if(k < 0) k += O;
        if(w < 0) w += P;        
        return host[i*N + j*O + k*P + w];                
    }
    float& operator()(array_index i, array_index j)    {
        assert(i < M);
        assert(j < N);                
        if(i < 0) i += M;        
        if(j < 0) j += N;        
        return host[i*N + j];                
    }
    float& operator()(array_index i, array_index j, array_index k)    {
        assert(i < M);
        assert(j < N);        
        assert(k < O);
        if(i < 0) i += M;        
        if(j < 0) j += N;
        if(k < 0) k += O;     
        return host[i*N+j*O+k];                
    }
    float& operator()(array_index i, array_index j, array_index k, array_index w)    {
        assert(i < M);
        assert(j < N);        
        assert(k < O);
        assert(w < P);
        if(i < 0) i += M;        
        if(j < 0) j += N;
        if(k < 0) k += O;
        if(w < 0) w += P;        
        return host[i*N+j*O+k*P+w];
    }
    
    GPUArrayView    __getitem(GPUIndex pos);    
    GPUArrayView    __getitem(array_index pos);

    GPUArrayView slice_view(array_index i, array_index j=0, array_index k=0);
    GPUArrayView slice_view(GPUIndex idx);

    void            __setitem(array_index pos, const float val) {         
        if(pos < 0) pos += M*N*O*P;
        host[pos] = val;         
    }
        
    array_index get_index(array_index i) { if(i < 0) i += M*N*O*P; return i; }
    array_index get_index(array_index i,array_index j) { 
        if(i < 0) i += M;
        if(j < 0) j += N;
        return i*N+j; 
    }
    array_index get_index(array_index i, array_index j, array_index k) { 
        if(i < 0) i += M;
        if(j < 0) j += N;
        if(k < 0) k += O;
        return i*N+j*O+k; 
    }
    array_index get_index(array_index i, array_index j, array_index k,array_index w) { 
        if(i < 0) i += M;
        if(j < 0) j += N;
        if(k < 0) k += O;
        if(w < 0) w += P;
        return i*N+j*O+k*P+w; 
    }

    void set_index(array_index i,float value) { 
        if(i < 0) i+= M;
        host[i] = value; 
    }
    void set_index(array_index i, array_index j, float value) { 
        if(i < 0) i += M;
        if(j < 0) j += N;
        host[get_index(i,j)] = value; 
    }
    void set_index(array_index i, array_index j, array_index k, float value) { 
        if(i < 0) i += M;
        if(j < 0) j += N;
        if(k < 0) k += O;
        host[get_index(i,j,k)] = value; 
    }
    void set_index(array_index i, array_index j, array_index k, array_index w, float value) { 
        if(i < 0) i += M;
        if(j < 0) j += N;
        if(k < 0) k += O;
        if(w < 0) w += P;    
        host[get_index(i,j,k,w)] = value; 
    }
    

    void print()  {
        download_host();               
        std::cout << "array[" << M << "," << N << "," << O << "," << P << "]" << std::endl;
        for(size_t w = 0; w < M*N*O*P; w++) {
            std::cout << host[w] << ",";
        }
        std::cout << std::endl;        
    }

// cublas

    // on device
    void copy_matrix_device(float * devptr, size_t sz) {
        assert(sz == size());
        cudaMemcpy(devPtr,devptr, sz*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    // on host
    void copy_matrix_host(float * hostptr, size_t sz) {
        assert(sz == size());
        cudaMemcpy(devPtr,hostptr, sz*sizeof(float), cudaMemcpyHostToDevice);
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
    void SetMatrixAsync(CudaStream& stream, float * matrix)    {
        int lda = M;
        int ldb = M;
        cublasStatus_t err = cublasSetMatrixAsync(M,N,sizeof(float),matrix,lda,devPtr,ldb,stream.stream);
        ASSERTOK(err);
    }
    float* GetMatrixAsync(CudaStream & stream, float * buffer)    {     
        int lda = M;
        int ldb = M;
        cublasStatus_t err = cublasGetMatrixAsync(M,N,sizeof(float),devPtr,lda,buffer,ldb,stream.stream);
        ASSERTOK(err);
        return buffer;
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
    GPUArray copy() const {
        GPUArray r(M,N,O,P);
        cudaMemcpy(r.devPtr,devPtr,size()*sizeof(float), cudaMemcpyDeviceToDevice);
        return r;
    }
    GPUArray clone() const { return copy(); } 
    GPUArray eval() const { return copy(); }     
};


//////////////////////////////////////////////////////////////////////////////////
// ArrayView 
// This is awkward.
// I want to change it to a rectangular window in memory 
// It should be able to map to linear algebra in a Tensor in 1 or 2-d.
//////////////////////////////////////////////////////////////////////////////////
struct GPUArrayView
{
    GPUArray    *array;
    array_index jrow;
    array_index krow;
    array_index wrow;
    

    GPUArrayView(GPUArray* array, array_index i, array_index j=0, array_index k=0) {
        this->array = array;
        if(i < 0) i += array->M;
        if(j < 0) j += array->N;
        if(k < 0) k += array->O;
        jrow = i*array->N;
        krow = j*array->O;
        wrow = k*array->P;        
    }
    float& operator[](array_index i) {        
        size_t index = i+jrow+krow+wrow;
        return (*array)[index];
    }
    float __getitem(array_index i) {
        size_t index = i+jrow+krow+wrow;
        return (*array)[index];
    }
    void __setitem(array_index i, float value) {
        size_t index = i+jrow+krow+wrow;
        (*array)[index] = value;
    }
};

GPUArrayView  GPUArray:: __getitem(GPUIndex pos) { 
    return GPUArrayView(this,pos.i,pos.j,pos.k); 
}
GPUArrayView  GPUArray:: __getitem(array_index pos) {     
    return GPUArrayView(this,pos); 
}

GPUArrayView GPUArray::slice_view(array_index i, array_index j, array_index k) {
    return GPUArrayView(this,i,j,k);
}
GPUArrayView GPUArray::slice_view(GPUIndex idx) {
    return GPUArrayView(this,idx.i,idx.j,idx.k);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPUVector (float)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct GPUVector
{
    GPUArray         *        vector;    
    std::shared_ptr<GPUArray> vector_ptr;

    GPUVector() {
        vector = nullptr;
    }
    GPUVector(size_t i)     {
        vector = new GPUArray(i);
        assert(vector != NULL);
        vector->zero();
        vector_ptr = std::shared_ptr<GPUArray>(vector);
    }
    GPUVector(const GPUVector & v)    {
        //vector = v.vector;                
        //vector_ptr = v.vector_ptr;
        vector = new GPUArray(*v.vector);
        vector_ptr = std::shared_ptr<GPUArray>(vector);
    }
    GPUVector(const GPUArray & a) {
        vector = new GPUArray(a);
        assert(vector != NULL);
        vector_ptr = std::shared_ptr<GPUArray>(vector);
    }
    GPUVector(const std::vector<float> & data)    {
        vector = new GPUArray(data,data.size());        
        assert(vector != NULL);
        vector_ptr = std::shared_ptr<GPUArray>(vector);
    }
    GPUVector(float * data, size_t length)    {
        vector = new GPUArray(data,length);        
        assert(vector != NULL);
        vector_ptr = std::shared_ptr<GPUArray>(vector);
    }
    ~GPUVector() = default;

    GPUVector copy() const {
        GPUVector r = vector->copy();
        return r;
    }
    GPUVector clone() const { return copy(); }     
    GPUVector eval() const  { return copy(); } 

    size_t size() const { return vector->size(); }
    void   zero() { vector->zero(); }
    void   ones() { vector->ones(); }
    void   random() { vector->randu(); }    
    void   fill(float v) { vector->fill(v); }
    bool   has_nans() { return vector->has_nans(); }

    size_t M() const  { return vector->M; }    

    float& operator[](array_index index) { 
        if(index <0) index += M();
        return (*vector)[index]; 
    }
    float  __getitem(array_index index) { 
        if(index < 0) index += M();
        return (*vector)[index]; 
    }
    void   __setitem(array_index index, float v) { 
        if(index < 0) index += M();
        (*vector)[index]=v; 
    }
    
    size_t get_index(array_index i) { 
        if(i < 0) i += M();
        return vector->get_index(i); 
    }
    void   set_index(array_index i, float value) { 
        if(i < 0) i += M();
        vector->host[i] = value;
    }

    void download_host() { vector->download_host(); }
    void upload_device() { vector->upload_device(); }
    

    void set_device(std::vector<float> & v)    {
        cublasStatus_t err = cublasSetVector(vector->size(),sizeof(float),v.data(),0,vector->devPtr,0);
        ASSERTOK(err);
    }
    void get_device(std::vector<float> & v)    {
        v.resize(vector->size());
        cublasStatus_t err = cublasGetVector(vector->size(),sizeof(float),vector->devPtr,0,v.data(),0);
        ASSERTOK(err);
    }
    void set_device(float * v, size_t len)    {
        assert(len <= vector->size());
        cublasStatus_t err = cublasSetVector(len,sizeof(float),v,0,vector->devPtr,0);
        ASSERTOK(err);
    }
    void get_device(float * v, size_t size)    {
        assert(size <= vector->size());
        cublasStatus_t err = cublasGetVector(size,sizeof(float),vector->devPtr,0,v,0);
        ASSERTOK(err);
    }
    void print() {
        vector->print();
    }


    GPUVector& operator = (const GPUVector & v) { 
        if(vector == nullptr) {
            vector = v.vector;
            vector_ptr = v.vector_ptr;
        }
        else if(size() == v.size()) {
            cudaMemcpy(vector->devPtr,v.vector->devPtr,size()*sizeof(float), cudaMemcpyDeviceToDevice);            
        }
        else {
            vector = v.vector;            
            vector_ptr.reset(vector);            
        }
        return *this;
    }
    GPUVector& operator = (const GPUArray & v) { 
        vector     = new GPUArray(v);
        vector_ptr.reset(vector);
        return *this;
    }

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
    // cross product only works on 3-d vectors.
    //GPUVector cross(const GPUVector & b);
    
};




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPUMatrix (float)
// It is really awkward and temporary until I decide to improve it.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct GPUMatrix
{
    GPUArray * matrix;
    std::shared_ptr<GPUArray> matrix_ptr;

//////////////////////
// Constructor
//////////////////////
    GPUMatrix() {
        matrix = nullptr;
    }
    GPUMatrix(const GPUArray & a) {
        matrix = new GPUArray(a);
        matrix_ptr = std::shared_ptr<GPUArray>(matrix);
    }
    GPUMatrix(const GPUMatrix & m) {
        //matrix = m.matrix;
        //matrix_ptr = m.matrix_ptr;
        matrix = new GPUArray(*m.matrix);
        assert(matrix != NULL);
        matrix_ptr = std::shared_ptr<GPUArray>(matrix);
    }
    GPUMatrix(size_t i, size_t j)    {
        matrix = new GPUArray(i,j);
        assert(matrix != NULL);
        matrix->zero();        
        matrix_ptr = std::shared_ptr<GPUArray>(matrix);
    }
    GPUMatrix(std::vector<float> & data, size_t i, size_t j)    {
        matrix = new GPUArray(data,i,j);
        assert(matrix != NULL);        
        matrix_ptr = std::shared_ptr<GPUArray>(matrix);        
    }
    GPUMatrix(float * p, size_t i, size_t j) {
        matrix = new GPUArray(p,i,j);
        assert(matrix != NULL);        
        matrix_ptr = std::shared_ptr<GPUArray>(matrix);
    }
    ~GPUMatrix() = default;

//////////////////////
// operators 
//////////////////////

    GPUMatrix& operator = (const GPUMatrix & m) {      
        matrix = new GPUArray(*m.matrix);
        matrix_ptr.reset(matrix);                                
        return *this;
    }
    
    GPUMatrix operator - () { return GPUMatrix(*matrix * -1.0f); }
    
    GPUMatrix operator + (const GPUMatrix & m);
    GPUMatrix operator - (const GPUMatrix & m);    
    GPUMatrix operator * (const GPUMatrix & m);
 
    // not standard linear algebra operators
    GPUMatrix operator / (const GPUMatrix & m) { return GPUMatrix(*matrix / *m.matrix); }
    GPUMatrix operator % (const GPUMatrix & m) { return GPUMatrix(*matrix % *m.matrix); }

    
    GPUMatrix operator + (float m) { return GPUMatrix(*matrix + m); }    
    GPUMatrix operator - (float m) { return GPUMatrix(*matrix - m); }    
    GPUMatrix operator * (float m) { return GPUMatrix(*matrix * m); }    
    GPUMatrix operator / (float m) { return GPUMatrix(*matrix / m); }
    GPUMatrix operator % (float m) { return GPUMatrix(*matrix % m); }



//////////////////////
/// rowwise
//////////////////////

    GPUMatrix operator * (const GPUVector & v)    {
        GPUMatrix r(*matrix);        
        for(size_t i = 0; i < matrix->M; i++)        {
            vector_mul_row(r.matrix->devPtr,(i*matrix->N),v.vector->devPtr,v.vector->M);
        }                
        return r;
    }    
    GPUMatrix operator + (const GPUVector & v)    {
        GPUMatrix r(*matrix);               
        for(size_t i = 0; i < matrix->M; i++)        {            
            vector_add_row(r.matrix->devPtr,(i*matrix->N),v.vector->devPtr,v.vector->M);
        }        
        return r;
    }
    GPUMatrix operator - (const GPUVector & v)    {
        GPUMatrix r(*matrix);
        for(size_t i = 0; i < matrix->M; i++)        {
            vector_sub_row(r.matrix->devPtr,(i*matrix->N),v.vector->devPtr,v.vector->M);
        }
        return r;
    }
    GPUMatrix operator / (const GPUVector & v)    {
        GPUMatrix r(*matrix);
        for(size_t i = 0; i < matrix->M; i++)        {
            vector_div_row(r.matrix->devPtr,(i*matrix->N),v.vector->devPtr,v.vector->M);
        }
        return r;
    }
    GPUMatrix operator % (const GPUVector & v)    {
        GPUMatrix r(*matrix);
        for(size_t i = 0; i < matrix->M; i++)        {
            vector_mod_row(r.matrix->devPtr,(i*matrix->N),v.vector->devPtr,v.vector->M);
        }
        return r;
    }

    GPUMatrix operator + (const GPUConst & v);
    GPUMatrix operator - (const GPUConst & v);
    GPUMatrix operator * (const GPUConst & v);
    GPUMatrix operator / (const GPUConst & v);
    GPUMatrix operator % (const GPUConst & v);

    GPUMatrix operator + (const GPUScalar & s);
    GPUMatrix operator - (const GPUScalar & s);
    GPUMatrix operator * (const GPUScalar & s);
    GPUMatrix operator / (const GPUScalar & s);
    GPUMatrix operator % (const GPUScalar & s);


//////////////////////
// rowwise matrix
//////////////////////

    // get diagonal
    // get upper triangle 
    // get lower triangle

    GPUVector row(array_index row) {
        if(row < 0) row += M();
        GPUVector r(N());
        for(size_t i = 0; i < N(); i++)
            r[i] = get2(row,i);
        return r;
    }
    GPUVector col(array_index col) {
        if(col < 0) col += N();
        GPUVector r(M());
        for(size_t i = 0; i < M(); i++)
            r[i] = get2(i,col);
        return r;
    }
    
    // check that m.size() == N
    GPUMatrix row_add(const GPUMatrix & m) {
        GPUMatrix r(*matrix);        
        for(size_t i = 0; i < matrix->M; i++)
            vector_add_row(r.matrix->devPtr,(i*matrix->N),m.matrix->devPtr,matrix->N);        
        return r;
    }
    GPUMatrix row_sub(const GPUMatrix & m) {
        GPUMatrix r(*matrix);        
        for(size_t i = 0; i < matrix->M; i++)
            vector_sub_row(r.matrix->devPtr,(i*matrix->N),m.matrix->devPtr,matrix->N);        
        return r;
    }
    GPUMatrix row_mul(const GPUMatrix & m) {
        GPUMatrix r(*matrix);        
        for(size_t i = 0; i < matrix->M; i++)
            vector_mul_row(r.matrix->devPtr,(i*matrix->N),m.matrix->devPtr,matrix->N);        
        return r;
    }
    GPUMatrix row_div(const GPUMatrix & m) {
        GPUMatrix r(*matrix);        
        for(size_t i = 0; i < matrix->M; i++)
            vector_div_row(r.matrix->devPtr,(i*matrix->N),m.matrix->devPtr,matrix->N);        
        return r;
    }
    GPUMatrix row_mod(const GPUMatrix & m) {
        GPUMatrix r(*matrix);        
        for(size_t i = 0; i < matrix->M; i++)
            vector_mod_row(r.matrix->devPtr,(i*matrix->N),m.matrix->devPtr,matrix->N);        
        return r;
    }


//////////////////////
// matmul
//////////////////////
    GPUMatrix matmul(const GPUMatrix & b, bool transa=false,bool transb=false, bool transc=false,float alpha=1.0,float beta=0.0);
    GPUMatrix mm(const GPUMatrix & b, bool transa=false,bool transb=false,bool transc = false,float alpha=1.0,float beta=0.0) { return matmul(b,transa,transb,transc,alpha,beta); }
    GPUVector matvec(const GPUVector & v, bool transa=false, float alpha = 1.0, float beta = 0.0);
    GPUVector mv(const GPUVector & v, bool transa=false,float alpha=1.0,float beta=0.0) { return matvec(v,transa,alpha,beta); }


//////////////////////
// index
//////////////////////
    size_t M() const { return matrix->M; }
    size_t N() const { return matrix->N; }


    float& operator()(array_index i, array_index j) {
        if(i < 0) i += M();
        if(j < 0) j += N();        
        return matrix->host[i*N()+j];
    }
    float& operator[](array_index index) { 
        if(index < 0) index += M()*N();
        return matrix->host[index]; 
    }
    
    GPUMatrixView __getitem(array_index index);

    void __setitem(array_index index, float value) {
        if(index < 0) index += M()*N();
        matrix->host[index] = value;        
    }
    float get2(array_index r, array_index c) {         
        if(r < 0) r += M();
        if(c < 0) c += N();
        return matrix->host[get_index(r,c)];
    }    
    void set2(array_index r, array_index c, float value) {
        if(r < 0) r += M();
        if(c < 0) c += N();
        matrix->host[get_index(r,c)] = value;        
    }

    size_t get_index(array_index r, array_index c) { 
        if(r < 0) r += M();
        if(c < 0) c += N();    
        return matrix->get_index(r,c); 
    }
    void   set_index(array_index r, array_index c, float value) { 
        if(r < 0) r += M();
        if(c < 0) c += N();    
        matrix->set_index(r,c,value); 
    }
    
    size_t rows() { return matrix->M; }
    size_t cols() { return matrix->N; }
    size_t size() const { return matrix->size(); }
    void   zero() { matrix->zero(); }
    void   ones() { matrix->ones(); }
    void   fill(const float val) { matrix->fill(val); }
    void   random() { matrix->randu(); }
    bool   has_nans() { return matrix->has_nans(); }

    // is it called rank?
    void swap_order() {
        array_index temp = matrix->M;
        matrix->M = matrix->N;
        matrix->N = temp;
    }
//////////////////////
// device/host    
//////////////////////
    void download_host() { matrix->download_host(); }
    void upload_device() { matrix->upload_device(); }
        
    void set_device(std::vector<float> & v)    {
        cublasStatus_t err = cublasSetVector(matrix->size(),sizeof(float),v.data(),0,matrix->devPtr,0);
        ASSERTOK(err);
    }
    void get_device(std::vector<float> & v)    {
        v.resize(matrix->size());
        cublasStatus_t err = cublasGetVector(matrix->size(),sizeof(float),matrix->devPtr,0,v.data(),0);
        ASSERTOK(err);
    }
    void set_device(float * v, size_t len)    {
        assert(len <= matrix->size());
        cublasStatus_t err = cublasSetVector(len,sizeof(float),v,0,matrix->devPtr,0);
        ASSERTOK(err);
    }
    void get_device(float * v, size_t size)    {
        assert(size <= matrix->size());
        cublasStatus_t err = cublasGetVector(size,sizeof(float),matrix->devPtr,0,v,0);
        ASSERTOK(err);
    }

//////////////////////
// utilities    
//////////////////////
    void print() {
        //matrix->print();
        download_host();
        print_dims();
        for(size_t i = 0; i < M(); i++) {
            for(size_t j = 0; j < N(); j++) 
            {
                std::cout << (*this)(i,j);
                if(j < (N()-1)) std::cout << ",";
            }            
            std::cout << std::endl;
        }        
    }
    void print_dims() const {
        std::cout << "Matrix(" << M() << "," << N() << ")" << std::endl;
    }

    void identity()  {     
        // identity only makes sense on square matrix.   
        assert(M() == N());
        size_t c = 0;
        download_host();
        fill(0);
        for(size_t i = 0; i < matrix->M; i++) {
            matrix->host[i*matrix->N + c++] = 1;
        }            
        upload_device();
    }

    GPUMatrix t();

    
    GPUMatrix copy() const {
        GPUMatrix r = matrix->copy();        
        return r;
    }
    GPUMatrix clone() const { return copy(); }     
    GPUMatrix eval() const { return copy(); } 

};


/////////////////////////////////////////////////
// MatrixView 
// Makes a window to index into it from Lua 
/////////////////////////////////////////////////
struct GPUMatrixView
{
    GPUMatrix *matrix;
    array_index jrow;
    
    GPUMatrixView(GPUMatrix * matrix, array_index i) {
        this->matrix = matrix;
        if(i < 0) i += matrix->M();
        jrow = i*matrix->N();        
    }
    float& operator[](array_index i) {
        if(i < 0) i += matrix->M();
        array_index index = i+jrow;
        return (*matrix)[index];
    }
    float __getitem(array_index i) {
        if(i < 0) i += matrix->M();
        array_index index = i+jrow;
        return (*matrix)[index];
    }
    void __setitem(array_index i, float value) {
        if(i < 0) i += matrix->M();
        array_index index = i+jrow;
        (*matrix)[index] = value;
    }    
};

GPUMatrixView GPUMatrix::__getitem(array_index index) 
{ 
    return GPUMatrixView(this,index); 
}




/////////////////////////////////////////////////
// vector maths
// I would prefer it is not this way at all 
// I am not even sure if all of these are useful as vector operators.
// But I generated them from the cuda math page.
/////////////////////////////////////////////////
float sum(const GPUArray & a)     {
    float p = vector_sum(a.devPtr,a.size());
    return p;
}
float prod(const GPUArray & a)     {
    float p = vector_prod(a.devPtr,a.size());
    return p;
}

float sum(const GPUVector & a)     {
    float p = vector_sum(a.vector->devPtr,a.size());
    return p;
}
float prod(const GPUVector & a)     {
    float p = vector_prod(a.vector->devPtr,a.size());
    return p;
}

float sum(const GPUMatrix & a)     {
    float p = vector_sum(a.matrix->devPtr,a.size());
    return p;
}
float prod(const GPUMatrix & a)     {
    float p = vector_prod(a.matrix->devPtr,a.size());
    return p;
}

GPUArray acos(const GPUArray & a)     {
    float * p = vector_acosf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector acos(const GPUVector & a)     {
    float * p = vector_acosf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix acos(const GPUMatrix & a)     {
    float * p = vector_acosf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray acosh(const GPUArray & a)     {
    float * p = vector_acoshf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector acosh(const GPUVector & a)     {
    float * p = vector_acoshf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix acosh(const GPUMatrix & a)     {
    float * p = vector_acoshf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray asin(const GPUArray & a)     {
    float * p = vector_asinf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector asin(const GPUVector & a)     {
    float * p = vector_asinf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix asin(const GPUMatrix & a)     {
    float * p = vector_asinf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray atan2(const GPUArray & a,const GPUArray & b)     {
    float * p = vector_atan2f(a.devPtr,b.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector atan2(const GPUVector & a,const GPUVector & b)     {
    float * p = vector_atan2f(a.vector->devPtr,b.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix atan2(const GPUMatrix & a,const GPUMatrix & b)     {
    float * p = vector_atan2f(a.matrix->devPtr,b.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray atan2(const GPUArray & a,const float b)     {    
    float * p = vector_atan2f_const(a.devPtr,b,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector atan2(const GPUVector & a,const float b)     {    
    float * p = vector_atan2f_const(a.vector->devPtr,b,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix atan2(const GPUMatrix & a,const float b)     {    
    float * p = vector_atan2f_const(a.matrix->devPtr,b,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray atan(const GPUArray & a)     {
    float * p = vector_atanf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector atan(const GPUVector & a)     {
    float * p = vector_atanf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix atan(const GPUMatrix & a)     {
    float * p = vector_atanf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray cbrt(const GPUArray & a)     {
    float * p = vector_cbrtf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector cbrt(const GPUVector & a)     {
    float * p = vector_cbrtf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix cbrt(const GPUMatrix & a)     {
    float * p = vector_cbrtf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray ceil(const GPUArray & a)     {
    float * p = vector_ceilf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector ceil(const GPUVector & a)     {
    float * p = vector_ceilf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix ceil(const GPUMatrix & a)     {
    float * p = vector_ceilf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray copysign(const GPUArray & a,const GPUArray & b)     {
    float * p = vector_copysignf(a.devPtr,b.devPtr,a.size());
    return GPUArray(p,a.size());
}


GPUArray cos(const GPUArray & a)     {
    float * p = vector_cosf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector cos(const GPUVector & a)     {
    float * p = vector_cosf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix cos(const GPUMatrix & a)     {
    float * p = vector_cosf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray cosh(const GPUArray & a)     {
    float * p = vector_coshf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector cosh(const GPUVector & a)     {
    float * p = vector_coshf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix cosh(const GPUMatrix & a)     {
    float * p = vector_coshf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray cospi(const GPUArray & a)     {
    float * p = vector_cospif(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector cospi(const GPUVector & a)     {
    float * p = vector_cospif(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix cospi(const GPUMatrix & a)     {
    float * p = vector_cospif(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray cyl_bessel_i0(const GPUArray & a)     {
    float * p = vector_cyl_bessel_i0f(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector cyl_bessel_i0(const GPUVector & a)     {
    float * p = vector_cyl_bessel_i0f(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix cyl_bessel_i0(const GPUMatrix & a)     {
    float * p = vector_cyl_bessel_i0f(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray cyl_bessel_i1(const GPUArray & a)     {
    float * p = vector_cyl_bessel_i1f(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector cyl_bessel_i1(const GPUVector & a)     {
    float * p = vector_cyl_bessel_i1f(a.vector->devPtr,a.size());
    return GPUVector(p,a.M()); 
}
GPUMatrix cyl_bessel_i1(const GPUMatrix & a)     {
    float * p = vector_cyl_bessel_i1f(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray erfc(const GPUArray & a)     {
    float * p = vector_erfcf(a.devPtr,a.size());
    return GPUArray(p,a.size());
}
GPUArray erfcinv(const GPUArray & a)     {
    float * p = vector_erfinvf(a.devPtr,a.size());
    return GPUArray(p,a.size());
}
GPUArray erfcx(const GPUArray & a)     {
    float * p = vector_erfcxf(a.devPtr,a.size());
    return GPUArray(p,a.size());
}
GPUArray erf(const GPUArray & a)     {
    float * p = vector_erff(a.devPtr,a.size());
    return GPUArray(p,a.size());
}
GPUArray erfinv(const GPUArray & a)     {
    float * p = vector_erff(a.devPtr,a.size());
    return GPUArray(p,a.size());
}


GPUArray exp10(const GPUArray & a)     {
    float * p = vector_exp10f(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector exp10(const GPUVector & a)     {
    float * p = vector_exp10f(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix exp10(const GPUMatrix & a)     {
    float * p = vector_exp10f(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}



GPUArray exp2(const GPUArray & a)     {
    float * p = vector_exp2f(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector exp2(const GPUVector & a)     {
    float * p = vector_exp2f(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());    
}
GPUMatrix exp2(const GPUMatrix & a)     {
    float * p = vector_exp2f(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}



GPUArray exp(const GPUArray & a)     {
    float * p = vector_expf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector exp(const GPUVector & a)     {
    float * p = vector_expf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix exp(const GPUMatrix & a)     {
    float * p = vector_expf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray expm1(const GPUArray & a)     {
    float * p = vector_expm1f(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector expm1(const GPUVector & a)     {
    float * p = vector_expm1f(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix expm1(const GPUMatrix & a)     {
    float * p = vector_expm1f(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray fabs(const GPUArray & a)     {
    float * p = vector_fabsf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector fabs(const GPUVector & a)     {
    float * p = vector_fabsf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix fabs(const GPUMatrix & a)     {
    float * p = vector_fabsf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray fdim(const GPUArray & a,const GPUArray & b)     {
    float * p = vector_fdimf(a.devPtr,b.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector fdim(const GPUVector & a,const GPUVector & b)     {
    float * p = vector_fdimf(a.vector->devPtr,b.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix fdim(const GPUMatrix & a,const GPUMatrix & b)     {
    float * p = vector_fdimf(a.matrix->devPtr,b.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray fdim(const GPUArray & a,const float b)     {    
    float * p = vector_fdimf_const(a.devPtr,b,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}


GPUArray fdivide(const GPUArray & a,const GPUArray & b)     {
    float * p = vector_fdividef(a.devPtr,b.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUArray fdivide(const GPUArray & a,const float b)     {
    float * p = vector_fdividef_const(a.devPtr,b,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}

GPUArray fdivide(const GPUArray & x,const GPUArray & y, const GPUArray & z)     {
    float * p = vector_fmaf(x.devPtr,y.devPtr,z.devPtr,x.size());
    return GPUArray(p,x.M,x.N,x.O,x.P);
}
GPUVector fdivide(const GPUVector & x,const GPUVector & y, const GPUVector & z)     {
    float * p = vector_fmaf(x.vector->devPtr,y.vector->devPtr,z.vector->devPtr,x.size());
    return GPUVector(p,x.M());
}
GPUMatrix fdivide(const GPUMatrix & x,const GPUMatrix & y, const GPUMatrix & z)     {
    float * p = vector_fmaf(x.matrix->devPtr,y.matrix->devPtr,z.matrix->devPtr,x.size());
    return GPUMatrix(p,x.M(),x.N());
}

GPUArray floor(const GPUArray & a)     {
    float * p = vector_floorf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector floor(const GPUVector & a)     {
    float * p = vector_floorf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix floor(const GPUMatrix & a)     {
    float * p = vector_floorf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray fma(const GPUArray & a,const GPUArray & b, const GPUArray &c)     {
    float * p = vector_fmaf(a.devPtr,b.devPtr,c.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector fma(const GPUVector & a,const GPUVector & b, const GPUVector &c)     {
    float * p = vector_fmaf(a.vector->devPtr,b.vector->devPtr,c.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix fma(const GPUMatrix & a,const GPUMatrix & b, const GPUMatrix &c)     {
    float * p = vector_fmaf(a.matrix->devPtr,b.matrix->devPtr,c.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray fmax(const GPUArray & a,const GPUArray & b)     {
    float * p = vector_fmaxf(a.devPtr,b.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUArray fmax(const GPUArray & a,const float b) {    
    float * p = vector_fmaxf_const(a.devPtr,b,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}


GPUVector fmax(const GPUVector & a,const GPUVector & b)     {
    float * p = vector_fmaxf(a.vector->devPtr,b.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUVector fmax(const GPUVector & a,const float b)     {    
    float * p = vector_fmaxf_const(a.vector->devPtr,b,a.size());
    return GPUVector(p,a.M());
}

GPUMatrix fmax(const GPUMatrix & a,const GPUMatrix & b)     {
    float * p = vector_fmaxf(a.matrix->devPtr,b.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}
GPUMatrix fmax(const GPUMatrix & a,const float b)     {    
    float * p = vector_fmaxf_const(a.matrix->devPtr,b,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray fmin(const GPUArray & a,const GPUArray & b)     {
    float * p = vector_fminf(a.devPtr,b.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUArray fmin(const GPUArray & a,const float b)     {
    float * p = vector_fminf_const(a.devPtr,b,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}

GPUVector fmin(const GPUVector & a,const GPUVector & b)     {
    float * p = vector_fminf(a.vector->devPtr,b.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUVector fmin(const GPUVector & a,const float b)     {    
    float * p = vector_fminf_const(a.vector->devPtr,b,a.size());
    return GPUVector(p,a.M());
}

GPUMatrix fmin(const GPUMatrix & a,const GPUMatrix & b)     {
    float * p = vector_fminf(a.matrix->devPtr,b.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}
GPUMatrix fmin(const GPUMatrix & a,const float b)     {    
    float * p = vector_fminf_const(a.matrix->devPtr,b,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray fmod(const GPUArray & a,const GPUArray & b)     {
    float * p = vector_fmodf(a.devPtr,b.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUArray fmod(const GPUArray & a,const float b)     {    
    float * p = vector_fmodf_const(a.devPtr,b,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}

GPUVector fmod(const GPUVector & a,const GPUVector & b)     {
    float * p = vector_fmodf(a.vector->devPtr,b.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUVector fmod(const GPUVector & a,const float b)     {    
    float * p = vector_fmodf_const(a.vector->devPtr,b,a.size());
    return GPUVector(p,a.M());
}

GPUMatrix fmod(const GPUMatrix & a,const GPUMatrix & b)     {
    float * p = vector_fmodf(a.matrix->devPtr,b.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}
GPUMatrix fmod(const GPUMatrix & a,const float b)     {
    float * p = vector_fmodf_const(a.matrix->devPtr,b,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray hypot(const GPUArray & a,const GPUArray & b)     {
    float * p = vector_hypotf(a.devPtr,b.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUArray hypot(const GPUArray & a,const float b)     {    
    float * p = vector_hypotf_const(a.devPtr,b,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}

GPUVector hypot(const GPUVector & a,const GPUVector & b)     {
    float * p = vector_hypotf(a.vector->devPtr,b.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUVector hypot(const GPUVector & a,const float b)     {    
    float * p = vector_hypotf_const(a.vector->devPtr,b,a.size());
    return GPUVector(p,a.M());
}


GPUMatrix hypot(const GPUMatrix & a,const GPUMatrix & b)     {
    float * p = vector_hypotf(a.matrix->devPtr,b.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}
GPUMatrix hypot(const GPUMatrix & a,const float b)     {    
    float * p = vector_hypotf_const(a.matrix->devPtr,b,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray ilogb(const GPUArray & a)     {
    float * p = vector_ilogbf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector ilogb(const GPUVector & a)     {
    float * p = vector_ilogbf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix ilogb(const GPUMatrix & a)     {
    float * p = vector_ilogbf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray j0(const GPUArray & a)     {
    float * p = vector_j0f(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector j0(const GPUVector & a)     {
    float * p = vector_j0f(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix j0(const GPUMatrix & a)     {
    float * p = vector_j0f(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray j1(const GPUArray & a)     {
    float * p = vector_j1f(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector j1(const GPUVector & a)     {
    float * p = vector_j1f(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix j1(const GPUMatrix & a)     {
    float * p = vector_j1f(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray jn(const GPUArray & a,int N)     {
    float * p = vector_jnf(a.devPtr,N,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector jn(const GPUVector & a,int N)     {
    float * p = vector_jnf(a.vector->devPtr,N,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix jn(const GPUMatrix & a,int N)     {
    float * p = vector_jnf(a.matrix->devPtr,N,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray ldexp(const GPUArray & a,int exp) {
    float * p = vector_ldexpf(a.devPtr,exp,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector ldexp(const GPUVector & a,int exp) {
    float * p = vector_ldexpf(a.vector->devPtr,exp,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix ldexp(const GPUMatrix & a,int exp) {
    float * p = vector_ldexpf(a.matrix->devPtr,exp,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray lgamma(const GPUArray & a) {
    float * p = vector_lgammaf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector lgamma(const GPUVector & a) {
    float * p = vector_lgammaf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix lgamma(const GPUMatrix & a) {
    float * p = vector_lgammaf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray log10(const GPUArray & a) {
    float * p = vector_log10f(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector log10(const GPUVector & a) {
    float * p = vector_log10f(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix log10(const GPUMatrix & a) {
    float * p = vector_log10f(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray log1p(const GPUArray & a) {
    float * p = vector_log1pf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector log1p(const GPUVector & a) {
    float * p = vector_log1pf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix log1p(const GPUMatrix & a) {
    float * p = vector_log1pf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray log2(const GPUArray & a) {
    float * p = vector_log2f(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector log2(const GPUVector & a) {
    float * p = vector_log2f(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix log2(const GPUMatrix & a) {
    float * p = vector_log2f(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray logb(const GPUArray & a) {
    float * p = vector_logbf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}

GPUArray nearbyint(const GPUArray & a) {
    float * p = vector_nearbyintf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}

GPUArray norm3(const GPUArray & x,const GPUArray & y, const GPUArray & z) {
    float * p = vector_norm3df(x.devPtr,y.devPtr,z.devPtr,x.size());
    return GPUArray(p,x.M,x.N,x.O,x.P);
}
GPUVector norm3(const GPUVector & x,const GPUVector & y, const GPUVector & z) {
    float * p = vector_norm3df(x.vector->devPtr,y.vector->devPtr,z.vector->devPtr,x.size());
    return GPUVector(p,x.M());
}
GPUMatrix norm3(const GPUMatrix & x,const GPUMatrix & y, const GPUMatrix & z) {
    float * p = vector_norm3df(x.matrix->devPtr,y.matrix->devPtr,z.matrix->devPtr,x.size());
    return GPUMatrix(p,x.M(),x.N());
}


GPUArray norm4(const GPUArray & x,const GPUArray & y, const GPUArray & z, const GPUArray & q) {
    float * p = vector_norm4df(x.devPtr,y.devPtr,z.devPtr,q.devPtr,x.size());
    return GPUArray(p,x.M,x.N,x.O,x.P);
}
GPUVector norm4(const GPUVector & x,const GPUVector & y, const GPUVector & z, const GPUVector & q) {
    float * p = vector_norm4df(x.vector->devPtr,y.vector->devPtr,z.vector->devPtr,q.vector->devPtr,x.size());
    return GPUVector(p,x.M());
}
GPUMatrix norm4(const GPUMatrix & x,const GPUMatrix & y, const GPUMatrix & z, const GPUMatrix & q) {
    float * p = vector_norm4df(x.matrix->devPtr,y.matrix->devPtr,z.matrix->devPtr,q.matrix->devPtr,x.size());
    return GPUMatrix(p,x.M(),x.N());
}


GPUArray normcdf(const GPUArray & a) {
    float * p = vector_normcdff(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector normcdf(const GPUVector & a) {
    float * p = vector_normcdff(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix normcdf(const GPUMatrix & a) {
    float * p = vector_normcdff(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray normcdfinv(const GPUArray & a) {
    float * p = vector_normcdfinvf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector normcdfinv(const GPUVector & a) {
    float * p = vector_normcdfinvf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix normcdfinv(const GPUMatrix & a) {
    float * p = vector_normcdfinvf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray norm(int dim, const GPUArray & a) {
    float * p = vector_normf(dim,a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector norm(int dim, const GPUVector & a) {
    float * p = vector_normf(dim,a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix norm(int dim, const GPUMatrix & a) {
    float * p = vector_normf(dim,a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray pow(const GPUArray & x,const GPUArray & y) {
    float * p = vector_powf(x.devPtr,y.devPtr,x.size());
    return GPUArray(p,x.M,x.N,x.O,x.P);
}
GPUArray pow(const GPUArray & x,const float  y) {    
    float * p = vector_powf_const(x.devPtr,y,x.size());
    return GPUArray(p,x.M,x.N,x.O,x.P);
}


GPUVector pow(const GPUVector & x,const GPUVector & y) {
    float * p = vector_powf(x.vector->devPtr,y.vector->devPtr,x.size());
    return GPUVector(p,x.M());
}
GPUVector pow(const GPUVector & x,const float  y) {    
    float * p = vector_powf_const(x.vector->devPtr,y,x.size());
    return GPUVector(p,x.M());
}


GPUMatrix pow(const GPUMatrix & x,const GPUMatrix & y) {
    float * p = vector_powf(x.matrix->devPtr,y.matrix->devPtr,x.size());
    return GPUMatrix(p,x.M(),x.N());
}
GPUMatrix pow(const GPUMatrix & x,const float  y) {    
    float * p = vector_powf_const(x.matrix->devPtr,y,x.size());
    return GPUMatrix(p,x.M(),x.N());
}


GPUArray rcbrt(const GPUArray & a) {
    float * p = vector_rcbrtf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector rcbrt(const GPUVector & a) {
    float * p = vector_rcbrtf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix rcbrt(const GPUMatrix & a) {
    float * p = vector_rcbrtf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray remainder(const GPUArray & x,const GPUArray & y) {
    float * p = vector_remainderf(x.devPtr,y.devPtr,x.size());
    return GPUArray(p,x.M,x.N,x.O,x.P);
}
GPUArray remainder(const GPUArray & x,const float y) {    
    float * p = vector_remainderf_const(x.devPtr,y,x.size());
    return GPUArray(p,x.M,x.N,x.O,x.P);
}

GPUArray rhypot(const GPUArray & x,const GPUArray & y) {
    float * p = vector_rhypotf(x.devPtr,y.devPtr,x.size());
    return GPUArray(p,x.M,x.N,x.O,x.P);
}
GPUArray rhypot(const GPUArray & x,const float y) {    
    float * p = vector_rhypotf_const(x.devPtr,y,x.size());
    return GPUArray(p,x.M,x.N,x.O,x.P);
}

GPUVector rhypot(const GPUVector & x,const GPUVector & y) {
    float * p = vector_rhypotf(x.vector->devPtr,y.vector->devPtr,x.size());
    return GPUVector(p,x.M());
}
GPUVector rhypot(const GPUVector & x,const float y) {    
    float * p = vector_rhypotf_const(x.vector->devPtr,y,x.size());
    return GPUVector(p,x.M());
}

GPUMatrix rhypot(const GPUMatrix & x,const GPUMatrix & y) {
    float * p = vector_rhypotf(x.matrix->devPtr,y.matrix->devPtr,x.size());
    return GPUMatrix(p,x.M(),x.N());
}
GPUMatrix rhypot(const GPUMatrix & x,const float y) {    
    float * p = vector_rhypotf_const(x.matrix->devPtr,y,x.size());
    return GPUMatrix(p,x.M(),x.N());
}


GPUArray rnorm3d(const GPUArray & x,const GPUArray & y, const GPUArray & z) {
    float * p = vector_rnorm3df(x.devPtr,y.devPtr,z.devPtr,x.size());
    return GPUArray(p,x.M,x.N,x.O,x.P);
}
GPUVector rnorm3d(const GPUVector & x,const GPUVector & y, const GPUVector & z) {
    float * p = vector_rnorm3df(x.vector->devPtr,y.vector->devPtr,z.vector->devPtr,x.size());
    return GPUVector(p,x.M());
}
GPUMatrix rnorm3d(const GPUMatrix & x,const GPUMatrix & y, const GPUMatrix & z) {
    float * p = vector_rnorm3df(x.matrix->devPtr,y.matrix->devPtr,z.matrix->devPtr,x.size());
    return GPUMatrix(p,x.M(),x.N());
}


GPUArray rnorm4d(const GPUArray & x,const GPUArray & y, const GPUArray & z, const GPUArray & q) {
    float * p = vector_rnorm4df(x.devPtr,y.devPtr,z.devPtr,q.devPtr,x.size());
    return GPUArray(p,x.M,x.N,x.O,x.P);
}
GPUVector rnorm4d(const GPUVector & x,const GPUVector & y, const GPUVector & z, const GPUVector & q) {
    float * p = vector_rnorm4df(x.vector->devPtr,y.vector->devPtr,z.vector->devPtr,q.vector->devPtr,x.size());
    return GPUVector(p,x.M());
}
GPUMatrix rnorm4d(const GPUMatrix & x,const GPUMatrix & y, const GPUMatrix & z, const GPUMatrix & q) {
    float * p = vector_rnorm4df(x.matrix->devPtr,y.matrix->devPtr,z.matrix->devPtr,q.matrix->devPtr,x.size());
    return GPUMatrix(p,x.M(),x.N());
}


GPUArray rnorm(int dim, const GPUArray & a) {
    float * p = vector_rnormf(dim,a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector rnorm(int dim, const GPUVector & a) {
    float * p = vector_rnormf(dim,a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix rnorm(int dim, const GPUMatrix & a) {
    float * p = vector_rnormf(dim,a.matrix->devPtr,a.size());
    return GPUArray(p,a.M(),a.N());
}


GPUArray rsqrt(const GPUArray & a) {
    float * p = vector_rsqrtf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector rsqrt(const GPUVector & a) {
    float * p = vector_rsqrtf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix rsqrt(const GPUMatrix & a) {
    float * p = vector_rsqrtf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray scalbln(const GPUArray & a, long int N) {
    float * p = vector_scalblnf(a.devPtr,N,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector scalbln(const GPUVector & a, long int N) {
    float * p = vector_scalblnf(a.vector->devPtr,N,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix scalbln(const GPUMatrix & a, long int N) {
    float * p = vector_scalblnf(a.matrix->devPtr,N,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray sin(const GPUArray & a) {
    float * p = vector_sinf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector sin(const GPUVector & a) {
    float * p = vector_sinf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix sin(const GPUMatrix & a) {
    float * p = vector_sinf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray sinh(const GPUArray & a) {
    float * p = vector_sinhf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector sinh(const GPUVector & a) {
    float * p = vector_sinhf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix sinh(const GPUMatrix & a) {
    float * p = vector_sinhf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray sinpi(const GPUArray & a) {
    float * p = vector_sinpif(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector sinpi(const GPUVector & a) {
    float * p = vector_sinpif(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix sinpi(const GPUMatrix & a) {
    float * p = vector_sinpif(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray sqrt(const GPUArray & a) {
    float * p = vector_sqrtf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector sqrt(const GPUVector & a) {
    float * p = vector_sqrtf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix sqrt(const GPUMatrix & a) {
    float * p = vector_sqrtf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray tan(const GPUArray & a) {
    float * p = vector_tanf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector tan(const GPUVector & a) {
    float * p = vector_tanf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix tan(const GPUMatrix & a) {
    float * p = vector_tanf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray tanh(const GPUArray & a) {
    float * p = vector_tanhf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector tanh(const GPUVector & a) {
    float * p = vector_tanhf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix tanh(const GPUMatrix & a) {
    float * p = vector_tanhf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray tgamma(const GPUArray & a) {
    float * p = vector_tgammaf(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector tgamma(const GPUVector & a) {
    float * p = vector_tgammaf(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix tgamma(const GPUMatrix & a) {
    float * p = vector_tgammaf(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}


GPUArray y0(const GPUArray & a) {
    float * p = vector_y0f(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector y0(const GPUVector & a) {
    float * p = vector_y0f(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix y0(const GPUMatrix & a) {
    float * p = vector_y0f(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray y1(const GPUArray & a) {
    float * p = vector_y1f(a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}
GPUVector y1(const GPUVector & a) {
    float * p = vector_y1f(a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}
GPUMatrix y1(const GPUMatrix & a) {
    float * p = vector_y1f(a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}

GPUArray yn(int N,const GPUArray & a) {
    float * p = vector_ynf(N,a.devPtr,a.size());
    return GPUArray(p,a.M,a.N,a.O,a.P);
}    
GPUVector yn(int N,const GPUVector & a) {
    float * p = vector_ynf(N,a.vector->devPtr,a.size());
    return GPUVector(p,a.M());
}    
GPUMatrix yn(int N,const GPUMatrix & a) {
    float * p = vector_ynf(N,a.matrix->devPtr,a.size());
    return GPUMatrix(p,a.M(),a.N());
}    


GPUArray sigmoid(const GPUArray & x)
{    
    float * p = vector_sigmoid(x.devPtr,x.size());
    GPUArray r(p,x.M,x.N,x.O,x.P);    
    return r;
}
GPUVector sigmoid(const GPUVector & x)
{    
    float * p = vector_sigmoid(x.vector->devPtr,x.size());
    GPUVector r(p,x.M());
    return r;
}
GPUMatrix sigmoid(const GPUMatrix & x)
{    
    float * p = vector_sigmoid(x.matrix->devPtr,x.size());    
    GPUMatrix r(p,x.M(),x.N());
    return r;
}


GPUArray sigmoid_grad(const GPUArray & x)
{    
    float * p = vector_sigmoid_grad(x.devPtr,x.size());
    GPUArray r(p,x.M,x.N,x.O,x.P);    
    return r;
}
GPUVector sigmoid_grad(const GPUVector & x)
{    
    float * p = vector_sigmoid_grad(x.vector->devPtr,x.size());
    GPUVector r(p,x.M());
    return r;
}
GPUMatrix sigmoid_grad(const GPUMatrix & x)
{    
    float * p = vector_sigmoid_grad(x.matrix->devPtr,x.size());
    GPUMatrix r(p,x.M(),x.N());
    return r;
}

GPUArray tanh_grad(const GPUArray & x)
{    
    float * p = vector_tanh_grad(x.devPtr,x.size());
    GPUArray r(p,x.M,x.N,x.O,x.P);    
    return r;
}
GPUVector tanh_grad(const GPUVector & x)
{    
    float * p = vector_tanh_grad(x.vector->devPtr,x.size());
    GPUVector r(p,x.M());
    return r;
}
GPUMatrix tanh_grad(const GPUMatrix & x)
{    
    float * p = vector_tanh_grad(x.matrix->devPtr,x.size());
    GPUMatrix r(p,x.M(),x.N());
    return r;
}


GPUArray relu(const GPUArray & x)
{    
    float * p = vector_relu(x.devPtr,x.size());
    GPUArray r(p,x.M,x.N,x.O,x.P);    
    return r;
}
GPUVector relu(const GPUVector & x)
{    
    float * p = vector_relu(x.vector->devPtr,x.size());
    GPUVector r(p,x.M());
    return r;
}
GPUMatrix relu(const GPUMatrix & x)
{    
    float * p = vector_relu(x.matrix->devPtr,x.size());
    GPUMatrix r(p,x.M(),x.N());
    return r;
}


GPUArray relu_grad(const GPUArray & x)
{    
    float * p = vector_relu_grad(x.devPtr,x.size());
    GPUArray r(p,x.M,x.N,x.O,x.P);    
    return r;
}
GPUVector relu_grad(const GPUVector & x)
{    
    float * p = vector_relu_grad(x.vector->devPtr,x.size());
    GPUVector r(p,x.M());
    return r;
}
GPUMatrix relu_grad(const GPUMatrix & x)
{    
    float * p = vector_relu_grad(x.matrix->devPtr,x.size());
    GPUMatrix r(p,x.M(),x.N());
    return r;
}


GPUArray softmax(const GPUArray & x)
{    
    float * p = vector_softmax(x.devPtr,x.size());
    GPUArray r(p,x.M,x.N,x.O,x.P);    
    return r;
}
GPUVector softmax(const GPUVector& x)
{    
    float * p = vector_softmax(x.vector->devPtr,x.size());
    GPUVector r(p,x.M());
    return r;
}
GPUMatrix softmax(const GPUMatrix& x)
{    
    float * p = vector_softmax(x.matrix->devPtr,x.size());
    GPUMatrix r(p,x.M(),x.N());
    return r;
}

// meh just return std::vector
std::vector<long long> llrint(const GPUArray & a)    {
    long long * p = vector_llrintf(a.devPtr,a.size());
    std::vector<long long> v;
    v.resize(a.size());
    cudaMemcpy(v.data(),p,a.size()*sizeof(long long), cudaMemcpyDeviceToHost);
    cudaFree(p);
    return v;
}
std::vector<long long> llround(const GPUArray & a){
    long long * p = vector_llroundf(a.devPtr,a.size());
    std::vector<long long> v;
    v.resize(a.size());
    cudaMemcpy(v.data(),p,a.size()*sizeof(long long), cudaMemcpyDeviceToHost);
    cudaFree(p);
    return v;
}
std::vector<long> lrint(const GPUArray & a)    {
    long *p = vector_lrintf(a.devPtr,a.size());
    std::vector<long> v;
    v.resize(a.size());
    cudaMemcpy(v.data(),p,a.size()*sizeof(long), cudaMemcpyDeviceToHost);
    cudaFree(p);
    return v;
}
std::vector<long> lround(const GPUArray & a){
    long * p = vector_lroundf(a.devPtr,a.size());
    std::vector<long> v;
    v.resize(a.size());
    cudaMemcpy(v.data(),p,a.size()*sizeof(long), cudaMemcpyDeviceToHost);
    cudaFree(p);
    return v;
}


// it is needed as some calls modify a pointer value.
// so it will work in Lua etc.
// Array can be used as GPU scalar 
struct Scalar {
    float value;
    Scalar(float x) : value(x) {}
};

//////////////////////////////////////////////////
// Blas Level1
//////////////////////////////////////////////////
// y = alpha*x[k] + y[j]
void axpy(Scalar& alpha, GPUVector & a, GPUVector & b, int incx=1, int incy=1)    {    
    cublasStatus_t     status; 
    status = cublasSaxpy(cublas->handle,a.vector->size(),&alpha.value,a.vector->devPtr, incx, b.vector->devPtr, incy);
    ASSERTOK(status);    
}
// y = alpha*x[k] + y[j]
void axpy(float _alpha, GPUVector & a, GPUVector & b, int incx=1, int incy=1)    {    
    Scalar alpha(_alpha);
    cublasStatus_t     status; 
    status = cublasSaxpy(cublas->handle,a.vector->size(),&alpha.value,a.vector->devPtr, incx, b.vector->devPtr, incy);
    ASSERTOK(status);    
}
// first index of greatest a[i]
int  amax(GPUVector & v, int incx=1)    {
    int result=-1;
    cublasStatus_t     status; 
    status = cublasIsamax(cublas->handle,v.vector->size(),v.vector->devPtr, incx, &result);
    ASSERTOK(status);
    return result;
}
// first index of least a[i]
int amin(GPUVector & v, int incx=1) {
    int result=-1;
    cublasStatus_t     status; 
    status = cublasIsamin(cublas->handle,v.vector->size(),v.vector->devPtr, incx, &result);
    ASSERTOK(status);    
    return result;
}
// x = sum(v)
Scalar  asum(GPUVector & v, int incx=1) {
    Scalar result(0);
    cublasStatus_t     status;     
    status = cublasSasum(cublas->handle,v.vector->size(),v.vector->devPtr, incx, &result.value);
    ASSERTOK(status);
    return result;
}
// normalize v and return value
Scalar nrm2(GPUVector & v, int incx=1) {        
    Scalar result(0);
    cublasStatus_t     status;     
    status = cublasSnrm2(cublas->handle,v.vector->size(),v.vector->devPtr, incx, &result.value);
    ASSERTOK(status);
    return result;
}

// dot x,y
Scalar dot(GPUVector & x,GPUVector & y, int incx=1,int incy=1) {
    Scalar result(0);
    cublasStatus_t     status; 
    status = cublasSdot(cublas->handle,x.vector->size(),x.vector->devPtr, incx, y.vector->devPtr,incy, &result.value);
    ASSERTOK(status);
    return result;
}
// rot x,y and return new vector
void rot(GPUVector & x, GPUVector & y, Scalar& cosine, Scalar& sinus, int incx=1,int incy=1) {    
    cublasStatus_t     status; 
    status = cublasSrot(cublas->handle,x.vector->size(),x.vector->devPtr, incx, y.vector->devPtr, incy, &cosine.value, &sinus.value);
    ASSERTOK(status);    
}
// make rotation matrix
void rotg(Scalar &a, Scalar &b, Scalar &cosine, Scalar &sinus) {                
    cublasStatus_t     status; 
    status = cublasSrotg(cublas->handle, &a.value, &b.value, &cosine.value, &sinus.value);
    ASSERTOK(status);
}
// rotate x,y and return new vector
void rotm(GPUVector & x,GPUVector & y, Scalar&  param, int incx=1,int incy=1) {    
    cublasStatus_t     status; 
    status = cublasSrotm(cublas->handle,x.vector->size(),x.vector->devPtr, incx, y.vector->devPtr, incy, &param.value);
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
    status = cublasSscal(cublas->handle, v.vector->size(), &alpha.value, v.vector->devPtr, incx);
    ASSERTOK(status);    
}
// scal v with alpha  and return new vector
void scal(GPUVector & v, float alpha,int incx=1) {    
    cublasStatus_t     status; 
    status = cublasSscal(cublas->handle, v.vector->size(), &alpha, v.vector->devPtr, incx);
    ASSERTOK(status);    
}

// swap this with v and return new vector
void swap(GPUVector & src, GPUVector & dst, int incx=1, int incy=1) {    
    cublasStatus_t     status; 
    status = cublasSswap(cublas->handle, src.vector->N, src.vector->devPtr, incx, dst.vector->devPtr, incy);    
}
// copy src to dst
void copy(GPUVector & src, GPUVector & dst, int incx=1,int incy=1)    {
    cublasStatus_t     status; 
    status = cublasScopy(cublas->handle,src.vector->size(),src.vector->devPtr, incx, dst.vector->devPtr, incy);
    ASSERTOK(status);    
}


/////////////////////////////////////
// Blas 2 and 3 
/////////////////////////////////////

// y = alpha * op(A)*x + beta * y
// op(A) = A if cublas_op_n 
// op(A) = A transpose if cublas_op_t 
// op(A) = A^H if cublas_op_h
void gbmv(cublasOperation_t trans, int m, int n, float alpha, GPUMatrix &A, int lda, int kl, int ku, GPUVector &x, float beta, GPUVector& y, int incx=1,int incy=1)
{         
    cublasStatus_t status = cublasSgbmv(cublas->handle,trans,m,n,kl,ku,&alpha,A.matrix->devPtr,lda,x.vector->devPtr,incx, &beta, y.vector->devPtr,incy);
    ASSERTOK(status);    
}

// r = alpha * op(A) * x + beta * y 
void gemv(cublasOperation_t trans, int m, int n, float alpha, GPUMatrix &A, int lda, GPUVector &x, float beta, GPUVector &y, int incx=1,int incy=1)
{            
    cublasStatus_t status = cublasSgemv(cublas->handle,trans,m,n,&alpha,A.matrix->devPtr,lda,x.vector->devPtr,incx,&beta,y.vector->devPtr,incy);
    ASSERTOK(status);    
}


// y = alpha * x * transpose(y) if  ger,geru
// y = alpha * x * H(y) if gerc
void ger(int m, int n, float alpha, GPUVector & x, GPUVector &y, GPUMatrix & A, int incx=1,int incy=1, int lda=-1)
{    
    if(lda == -1) lda = A.matrix->M;        
    cublasStatus_t status = cublasSger(cublas->handle,m,n,&alpha,x.vector->devPtr,incx,y.vector->devPtr,incy,A.matrix->devPtr,lda);
    ASSERTOK(status);    
}
// y = alpha * A * x + beta * y 
void sbmv(cublasFillMode_t uplo, int n, int k, float alpha, GPUMatrix & A, int lda,GPUVector &x, float beta, GPUVector & y, int incx=1, int incy=1)
{        
    cublasStatus_t status = cublasSsbmv(cublas->handle,uplo,n,k,&alpha,A.matrix->devPtr,lda,x.vector->devPtr,incx,&beta,y.vector->devPtr,incy);
    ASSERTOK(status);    
}

// A = alpha*x*tranpose(x) + A 
void spr(cublasFillMode_t uplo, int n, const float alpha, GPUVector & v, GPUMatrix & AP, int incx=1)
{    
    cublasStatus_t status = cublasSspr(cublas->handle,uplo,n,&alpha,v.vector->devPtr,incx, AP.matrix->devPtr);
    ASSERTOK(status);    
}

// A = alpha*(x*tranpose(y) + y*transpose(x)) + A
void spr2(cublasFillMode_t uplo, int n, const float alpha, GPUVector & x, GPUVector &y, GPUMatrix & AP, int incx=1, int incy=1)
{    
    cublasStatus_t status = cublasSspr2(cublas->handle,uplo,n,&alpha,x.vector->devPtr,incx,y.vector->devPtr,incy,AP.matrix->devPtr);
    ASSERTOK(status);    
}

// y = alpha*A*x + beta*y 
void symv(cublasFillMode_t uplo, int n, float alpha, GPUMatrix & A, int lda, GPUVector &x, float beta, GPUVector &y, int incx=1,int incy=1)
{    
    cublasStatus_t status = cublasSsymv(cublas->handle,uplo,n,&alpha,A.matrix->devPtr,lda,x.vector->devPtr,incx,&beta,y.vector->devPtr,incy);
    ASSERTOK(status);    
}

// A = alpha*x*tranpose(x) + A
void syr(cublasFillMode_t uplo, int n, float alpha, GPUVector &x, GPUMatrix &A, int lda, int incx=1)
{        
    cublasStatus_t status = cublasSsyr(cublas->handle, uplo, n, &alpha, x.vector->devPtr, incx, A.matrix->devPtr, lda);
    ASSERTOK(status);    
}

// A = alpha*(x*transpose(y) + y*transpose(x)) + A
void syr2(cublasFillMode_t uplo, float alpha, GPUVector & x, GPUVector & y, GPUMatrix & A, int lda, int incx=1,int incy=1)
{    
    cublasStatus_t status = cublasSsyr2(cublas->handle,uplo,A.matrix->M*A.matrix->N, &alpha,  x.vector->devPtr,incx, y.vector->devPtr,incy, y.vector->devPtr,lda );
    ASSERTOK(status);    
}

// op(A)*x = b
void tbmv(cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, GPUMatrix & A, int lda, GPUVector &x, int incx=1)
{    
    cublasStatus_t status = cublasStbmv(cublas->handle, uplo, trans, diag,n,k,A.matrix->devPtr,lda, x.vector->devPtr,incx);
    ASSERTOK(status);    
}

// b = op(A)x
void tbsv(cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int k, GPUMatrix & A, int lda, GPUVector & x, int incx=1)
{    
    cublasStatus_t status = cublasStbsv(cublas->handle,uplo,trans,diag,A.matrix->M*A.matrix->N,k,A.matrix->devPtr,lda,x.vector->devPtr,incx);
    ASSERTOK(status);    
}

// x = op(A)x
void trmv(cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, GPUMatrix & A, int lda, GPUVector & x, int incx=1)
{    
    cublasStatus_t status = cublasStrmv(cublas->handle,uplo,trans,diag,n,A.matrix->devPtr,lda,x.vector->devPtr,incx);
    ASSERTOK(status);    
}

// op(A)*x = b
void trsv(cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, GPUMatrix & A, int lda, GPUVector & x, int incx=1)
{        
    cublasStatus_t status = cublasStrsv(cublas->handle,uplo,trans,diag,n,A.matrix->devPtr,lda,x.vector->devPtr,incx);
    ASSERTOK(status);    
}

// general matrix multiply C=alpha*AB + beta*C
void gemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, GPUMatrix &A, int lda,  GPUMatrix &B, int ldb, float beta,  GPUMatrix &C, int ldc)
{       
    cublasStatus_t status = cublasSgemm(cublas->handle,transa,transb,m,n,k,&alpha,A.matrix->devPtr,lda,B.matrix->devPtr,ldb,&beta,C.matrix->devPtr,ldc);
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
    cublasStatus_t status = cublasSsymm(cublas->handle, side, uplo, m,n, &alpha, A.matrix->devPtr,lda,B.matrix->devPtr,ldb,&beta,C.matrix->devPtr,ldc);
    ASSERTOK(status);
}

// C = alpha * op(A) * op_transpose(A) + beta * C
// symetric rank update
void syrk(cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, GPUMatrix &A, int lda, float beta, GPUMatrix &C, int ldc)
{
    if(lda == -1) lda = A.matrix->M;
    if(ldc == -1) ldc = C.matrix->M;
    cublasStatus_t status = cublasSsyrk(cublas->handle,uplo,trans,n,k,&alpha,A.matrix->devPtr,lda,&beta,C.matrix->devPtr,ldc);
    ASSERTOK(status);
}

// C = alpha * (op(A) * op_transpose(B)) + op(B)+op_transposed(A) + beta*C 
void syr2k(cublasFillMode_t uplo, cublasOperation_t trans, float alpha, int n, int k, GPUMatrix & A, int lda, GPUMatrix & B, int ldb, float beta, GPUMatrix & C, int ldc)
{
    cublasStatus_t status = cublasSsyr2k(cublas->handle,uplo,trans,n,k,&alpha, A.matrix->devPtr,lda,B.matrix->devPtr,ldb,&beta,C.matrix->devPtr,ldc);
    ASSERTOK(status);
}
// C = alpha * op(A) * op_transposed(B) + beta * C
// symetric rank update variation
void syrkx(cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, float alpha, GPUMatrix & A, int lda, GPUMatrix & B, int ldb, float beta, GPUMatrix &C, int ldc)
{
    cublasStatus_t status = cublasSsyrkx(cublas->handle,uplo,trans,n,k,&alpha,A.matrix->devPtr,lda,B.matrix->devPtr,ldb,&beta,C.matrix->devPtr,ldc);
    ASSERTOK(status);
}

// if side == left C = alpha * op(A)*B 
// else C = alpha*B*op(A)
void trmm(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, float alpha, GPUMatrix & A, int lda, GPUMatrix & B, int ldb, GPUMatrix & C, int ldc)
{
    cublasStatus_t status = cublasStrmm(cublas->handle,side,uplo,trans,diag,m,n,&alpha,A.matrix->devPtr,lda,B.matrix->devPtr,ldb,C.matrix->devPtr,ldc);
    ASSERTOK(status);
}

// if side == left op(A)X = alpha * B
// else X * op(A) = alpha * B
void trsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n,float alpha, GPUMatrix & A, int lda, GPUMatrix & B, int ldb)
{
    cublasStatus_t status = cublasStrsm(cublas->handle,side,uplo,trans,diag,m,n,&alpha,A.matrix->devPtr,lda,B.matrix->devPtr,ldb);
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
    cublasStatus_t status = cublasSgeam(cublas->handle,transa,transb,m,n,&alpha,A.matrix->devPtr,lda,&beta,B.matrix->devPtr,ldb,C.matrix->devPtr,ldc);
    ASSERTOK(status);
}

void dgmm(cublasSideMode_t mode, int m, int n, GPUMatrix & A, int lda, GPUVector &x, GPUMatrix & C, int ldc, int incx=1)
{    
    cublasStatus_t status = cublasSdgmm(cublas->handle,mode,m,n,A.matrix->devPtr,lda, x.vector->devPtr,incx, C.matrix->devPtr, ldc);
    ASSERTOK(status);
}

void gemmEx(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, void* A, cudaDataType_t Atype, int lda, const void * B, cudaDataType_t Btype, int ldb, float beta, void * C, cudaDataType_t Ctype, int ldc)
{
    cublasStatus_t status = cublasSgemmEx(cublas->handle,transa,transb,m,n,k,&alpha,A,Atype,lda,B,Btype,ldb,&beta,C,Ctype,ldc);
    ASSERTOK(status);
}


///////////////////////////////////////////////
// Matrix
///////////////////////////////////////////////
// because it is in row major it uses identity C^t = A^t * B^t 
// C is transposed to row major in driver
GPUMatrix GPUMatrix::matmul(const GPUMatrix & b,bool transa,bool transb,bool transc, float alpha,float beta) {     
    GPUMatrix A(*this),B(b),C(matrix->M,b.matrix->N);
    cublasOperation_t ta = transa? CUBLAS_OP_T:CUBLAS_OP_N;
    cublasOperation_t tb = transb? CUBLAS_OP_T:CUBLAS_OP_N;    
    int m = A.M();
    int n = A.N();
    int k = B.N();
    gemm(ta,tb,n,m,k,alpha,B,n,A,k,beta,C,n);                        
    if(transa) C.matrix->M = b.matrix->N;
    if(transb) C.matrix->N = matrix->M;
    if(transc==true) C = C.t();
    return C;
}

GPUVector GPUMatrix::matvec(const GPUVector & v,bool transa, float alpha, float beta ) {
    GPUMatrix A(*this);
    GPUVector B(v);
    GPUVector R(B.size());
    int m = A.M();
    int n = A.N();
    cublasOperation_t ta = transa? CUBLAS_OP_N:CUBLAS_OP_T;
    gemv(ta,n,m,alpha,A,n,B,beta,R);
    return R;
}

// C^T = B^T * A^T
// assumes a and b are in row major order
GPUMatrix matmul(const GPUMatrix & a,const GPUMatrix & b, float alpha=1.0, float beta=0.0) {         
    assert(a.matrix->N == b.matrix->M);    
    GPUMatrix A(a),B(b),C(a.matrix->M,b.matrix->N);
    cublasOperation_t ta = CUBLAS_OP_N;
    cublasOperation_t tb = CUBLAS_OP_N;      
    int m = A.M();
    int k = A.N();
    int n = B.N();    
    gemm(ta,tb,n,m,k,alpha,B,n,A,k,beta,C,n);                        
    return C;    
}
// C^T = A*B
// A and B are in column major order
GPUMatrix matmulTT(const GPUMatrix & a,const GPUMatrix & b, float alpha=1.0, float beta=0.0) {         
    assert(a.matrix->N == b.matrix->M);
    GPUMatrix A(a),B(b);
    cublasOperation_t ta = CUBLAS_OP_T;
    cublasOperation_t tb = CUBLAS_OP_T;  
    GPUMatrix C(a.matrix->M,b.matrix->N);
    A.swap_order();
    B.swap_order();     
    int m = A.M();
    int k = A.N();
    int n = B.N(); 
    gemm(ta,tb,m,n,k,alpha,A,k,B,n,beta,C,k);                        
    return C;    
}

// C^T = B^T*A
// B is in column major order
GPUMatrix matmulTN(const GPUMatrix & a,const GPUMatrix & b, float alpha=1.0, float beta=0.0) {         
    assert(a.matrix->M == b.matrix->N);
    GPUMatrix A(a),B(b);
    cublasOperation_t ta = CUBLAS_OP_T;
    cublasOperation_t tb = CUBLAS_OP_N;           
    GPUMatrix C(a.matrix->M,b.matrix->N);
    A.swap_order();     
    int m = A.M();
    int k = A.N();
    int n = B.N(); 
    gemm(ta,tb,n,m,k,alpha,B,n,A,k,beta,C,n);                        
    return C;        
}
// C^T = B*A^T
// A is in column major order
GPUMatrix matmulNT(const GPUMatrix & a,const GPUMatrix & b, float alpha=1.0, float beta=0.0) {         
    assert(a.matrix->M == b.matrix->N);
    GPUMatrix A(a),B(b);
    cublasOperation_t ta = CUBLAS_OP_N;
    cublasOperation_t tb = CUBLAS_OP_T;            
    GPUMatrix C(a.matrix->M,b.matrix->N);    
    B.swap_order();
    int m = A.M();
    int k = A.N();
    int n = B.N(); 
    gemm(ta,tb,n,m,k,alpha,B,n,A,k,beta,C,n);                        
    return C;    
}

GPUMatrix matmul_cuda(const GPUMatrix & a,const GPUMatrix & b)
{
    float * p = matrix_multiply(a.matrix->devPtr, b.matrix->devPtr, a.M(),a.N(),b.N());
    return GPUMatrix(p,a.M(),b.N());
}




GPUVector matvec(const GPUMatrix & a, const GPUVector & v,bool transa=false, float alpha=1.0, float beta=0.0) {
    GPUMatrix A(a);
    GPUVector B(v);
    GPUVector R(B.size());
    int m = A.M();
    int n = A.N();
    cublasOperation_t ta = transa? CUBLAS_OP_T:CUBLAS_OP_N;    
    gemv(CUBLAS_OP_N,m,n,alpha,A,m,B,beta,R);
    return R;
}

GPUMatrix GPUMatrix::t() {
    GPUMatrix r(*this);        
    r.swap_order();
    int m = M();
    int k = N();
    int n = r.N();
    geam(CUBLAS_OP_T,CUBLAS_OP_N,n,m,1.0,*this,n,*this,m,0.0,r,m);    
    return r;
}

GPUMatrix transpose(GPUMatrix & a){
    return a.t();
}

GPUMatrix GPUMatrix::operator + (const GPUMatrix & b) {         
    assert(matrix->M == b.matrix->M && matrix->N == b.matrix->N);
    GPUMatrix r(M(),N());
    int m = M();
    int k = N();    
    int n = N();
    geam(CUBLAS_OP_N,CUBLAS_OP_N,n,m,1.0,b,n,*this,k,1.0,r,n);
    return r;
}

GPUMatrix GPUMatrix::operator - (const GPUMatrix & b) { 
    assert(matrix->M == b.matrix->M && matrix->N == b.matrix->N);
    GPUMatrix r(M(),N());
    int m = M();
    int k = N();    
    int n = N();
    geam(CUBLAS_OP_N,CUBLAS_OP_N,n,m,-1.0,b,n,*this,k,1.0,r,n);
    return r;
}

GPUMatrix GPUMatrix::operator * (const GPUMatrix & m) 
{     
    float * p = matrix_hadamard(matrix->devPtr,m.matrix->devPtr,M(),N(),m.N());
    return GPUMatrix(p,M(),N());    
}

GPUMatrix hadamard_product(const GPUMatrix & a, const GPUMatrix & b) {
    float * p = matrix_hadamard(a.matrix->devPtr,b.matrix->devPtr,a.M(),a.N(),b.N());
    return GPUMatrix(p,a.M(),a.N());
}


GPUMatrix GPUMatrix::operator + (const GPUConst & v) {
    float *p = vector_add_scalar(matrix->devPtr,v.devPtr,size());
    return GPUMatrix(p,M(),N());
}
GPUMatrix GPUMatrix::operator - (const GPUConst & v) {
    float *p = vector_sub_scalar(matrix->devPtr,v.devPtr,size());
    return GPUMatrix(p,M(),N());
}
GPUMatrix GPUMatrix::operator * (const GPUConst & v) {
    float *p = vector_mul_scalar(matrix->devPtr,v.devPtr,size());
    return GPUMatrix(p,M(),N());
}
GPUMatrix GPUMatrix::operator / (const GPUConst & v) {
    float *p = vector_div_scalar(matrix->devPtr,v.devPtr,size());
    return GPUMatrix(p,M(),N());
}
GPUMatrix GPUMatrix::operator % (const GPUConst & v) {
    float *p = vector_mod_scalar(matrix->devPtr,v.devPtr,size());
    return GPUMatrix(p,M(),N());
}
GPUMatrix GPUMatrix::operator + (const GPUScalar & s) {
    float *p = vector_add_scalar(matrix->devPtr,s.devPtr,size());
    return GPUMatrix(p,M(),N());
}
GPUMatrix GPUMatrix::operator - (const GPUScalar & s) {
    float *p = vector_sub_scalar(matrix->devPtr,s.devPtr,size());
    return GPUMatrix(p,M(),N());
}
GPUMatrix GPUMatrix::operator * (const GPUScalar & s) {
    float *p = vector_mul_scalar(matrix->devPtr,s.devPtr,size());
    return GPUMatrix(p,M(),N());
}
GPUMatrix GPUMatrix::operator / (const GPUScalar & s) {
    float *p = vector_div_scalar(matrix->devPtr,s.devPtr,size());
    return GPUMatrix(p,M(),N());
}
GPUMatrix GPUMatrix::operator % (const GPUScalar & s) {
    float *p = vector_mod_scalar(matrix->devPtr,s.devPtr,size());
    return GPUMatrix(p,M(),N());
}







///////////////////////////////////////////////
// Vector
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
    status = cublasSdgmm(cublas->handle,CUBLAS_SIDE_RIGHT,1, size(), vector->devPtr, 1, y.vector->devPtr, 1, r.vector->devPtr,1);
    return r;
}
GPUVector GPUVector::operator / (GPUVector & y)    {    
    GPUVector       r(*vector / *y.vector);
    return r;
}
GPUVector GPUVector::operator % (GPUVector & y)    {    
    GPUVector       r(*vector % *y.vector);    
    return r;
}

GPUVector GPUVector::operator + (float v)    {    
    GPUVector A(*this), B(size());
    B.vector->fill(v);    
    axpy(1.0f,A,B);
    return B;
}
GPUVector GPUVector::operator - (float v)    {
    GPUVector A(*this), B(size());
    B.vector->fill(v);    
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
    B.vector->fill(v);
    GPUVector C(size());
    *C.vector = *A.vector % *B.vector;
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
    float *p = vector_add_scalar(vector->devPtr,v.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator - (const GPUConst & v){
    float *p = vector_sub_scalar(vector->devPtr,v.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator * (const GPUConst & v){
    float *p = vector_mul_scalar(vector->devPtr,v.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator / (const GPUConst & v){
    float *p = vector_div_scalar(vector->devPtr,v.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator % (const GPUConst & v){
    float *p = vector_mod_scalar(vector->devPtr,v.devPtr,size());
    return GPUVector(p,size());
}

GPUVector GPUVector::operator + (const GPUScalar & s){
    float *p = vector_add_scalar(vector->devPtr,s.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator - (const GPUScalar & s){
    float *p = vector_sub_scalar(vector->devPtr,s.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator * (const GPUScalar & s){
    float *p = vector_mul_scalar(vector->devPtr,s.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator / (const GPUScalar & s){
    float *p = vector_div_scalar(vector->devPtr,s.devPtr,size());
    return GPUVector(p,size());
}
GPUVector GPUVector::operator % (const GPUScalar & s){
    float *p = vector_mod_scalar(vector->devPtr,s.devPtr,size());
    return GPUVector(p,size());
}


/*
    std::vector<int8_t> to_int8() {
        std::vector<int8_t> r;
        r.resize(size());
        download_host();
        for(size_t i = 0; i < size(); i++) r[i] = (int8_t)host[i];
        return r;
    }
    std::vector<uint8_t> to_uint8() {
        std::vector<uint8_t> r;
        r.resize(size());
        download_host();
        for(size_t i = 0; i < size(); i++) r[i] = (uint8_t)host[i];
        return r;
    }
    std::vector<int16_t> to_int16() {
        std::vector<int16_t> r;
        r.resize(size());
        download_host();
        for(size_t i = 0; i < size(); i++) r[i] = (int16_t)host[i];
        return r;
    }
    std::vector<uint16_t> to_uint16() {
        std::vector<uint16_t> r;
        r.resize(size());
        download_host();
        for(size_t i = 0; i < size(); i++) r[i] = (uint16_t)host[i];
        return r;
    }
    std::vector<int32_t> to_int32() {
        std::vector<int32_t> r;
        r.resize(size());
        download_host();
        for(size_t i = 0; i < size(); i++) r[i] = (int32_t)host[i];
        return r;
    }
    std::vector<uint32_t> to_uint32() {
        std::vector<uint32_t> r;
        r.resize(size());
        download_host();
        for(size_t i = 0; i < size(); i++) r[i] = (uint32_t)host[i];
        return r;
    }

    void from_vector(std::vector<float> & v)
    {
        if(devPtr) cudaFree(devPtr);
        if(host) free(host);
        M = v.size();
        N = 1;
        cudaError_t err = cudaMalloc((void**)&devPtr,M*N*O*P*sizeof(float));
        assert(err == cudaSuccess);        
        host = (float*)calloc(M*N*O*P,sizeof(float));
        assert(host != NULL);    
        memcpy(host,v.data(),v.size()*sizeof(float));
        cudaMemcpy(devPtr,host,M*N*O*P*sizeof(float),cudaMemcpyHostToDevice);
    }
    void to_vector(std::vector<float> & v) {
        v.resize(M*N*O*P);
        download_host();
        memcpy(v.data(),host,M*N*O*P*sizeof(float));        
    }

*/