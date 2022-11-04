#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cfloat.h>

typedef float _Complex scomplex;
typedef double _Complex dcomplex;

typedef struct
{
    void * ptr;
    int    n_elements;
    int    e_size;
}
CPUMemory;


typedef enum {
    BV_AND,
    BV_OR,
    BV_XOR,
    BV_NAND,
    BV_NOR,
    BV_NXOR,
}
BV_OP;

typedef struct {
    uint8_t * bits;
    int       size;
    int       num_bits;
}
BitVector;


int sizeof_float();
int sizeof_double();
int sizeof_scomplex();
int sizeof_dcomplex();
int sizeof_byte();
int sizeof_word();
int sizeof_dword();
int sizeof_qword();
int sizeof_int8();
int sizeof_int16();
int sizeof_int32();
int sizeof_int64();

CPUMemory* cm_new(int num, int element_size);
void       cm_free(CPUMemory * mem);

float*  cm_get_float(CPUMemory * p, int index);
double* cm_get_doublle(CPUMemory * p, int index);
scomplex cm_get_scomplex(CPUMemory * p, int index);
dcomplex cm_get_domplex(CPUMemory * p, int index);
uint8_t* cm_get_byte(CPUMemory * p, int index);
int16_t* cm_get_word(CPUMemory * p, int index);
int32_t* cm_get_dword(CPUMemory * p, int index);
int64_t* cm_get_qword(CPUMemory * p, int index);
int8_t*  cm_get_int8(CPUMemory * p, int index);
int16_t* cm_get_int16(CPUMemory * p, int index);
int32_t* cm_get_int32(CPUMemory * p, int index);
int64_t* cm_get_int64(CPUMemory * p, int index);


void cm_set_byte(CPUMemory * p, int index, uint8_t value);
void cm_set_word(CPUMemory * p, int index, uint16_t value);
void cm_set_dword(CPUMemory * p, int index, uint32_t value);
void cm_set_qword(CPUMemory * p, int index, uint64_t value);
void cm_set_float(CPUMemory * p, int index, float value);
void cm_set_double(CPUMemory * p, int index, double value);
void cm_set_scomplex(CPUMemory * p, int index, scomplex value);
void cm_set_dcomplex(CPUMemory * p, int index, dcomplex value);

CPUMemory* cm_new(int num, int element_size) {
    CPUMemory * p = (CPUMemory*)calloc(1,sizeof(CPUMemory));
    assert(p!=NULL);
    p->ptr = calloc(num,element_size);
    assert(p->ptr) != NULL;
    p->n_elements = num;
    p->e_size     = element_size;
    return p;
}
void cm_free(CPUMemory * mem) {
    assert(mem != NULL);
    if(mem->ptr) free(mem->ptr);
    free(mem);
}
float  cm_get_float(CPUMemory * p, int index) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    return ((float*)p->ptr)[index];
}
double  cm_get_double(CPUMemory * p, int index) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    return ((double*)p->ptr)[index];
}
scomplex  cm_get_scomplex(CPUMemory * p, int index) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    return ((scomplex*)p->ptr)[index];
}
dcomplex  cm_get_dcomplex(CPUMemory * p, int index) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    return ((dcomplex*)p->ptr)[index];
}

uint8_t cm_get_byte(CPUMemory *p, int index) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    return ((uint8_t*)p->ptr)[index];
}
uint16_t cm_get_word(CPUMemory *p, int index) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    return ((uint16_t*)p->ptr)[index];
}
uint32_t cm_get_dword(CPUMemory *p, int index) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    return ((uint32_t*)p->ptr)[index];
}
uint64_t cm_get_qword(CPUMemory *p, int index) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    return ((uint64_t*)p->ptr)[index];
}

int8_t cm_get_int8(CPUMemory *p, int index) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    return ((int8_t*)p->ptr)[index];
}
int16_t cm_get_int16(CPUMemory *p, int index) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    return ((int16_t*)p->ptr)[index];
}
int32_t cm_get_int32(CPUMemory *p, int index) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    return ((int32_t*)p->ptr)[index];
}
int64_t cm_get_int64(CPUMemory *p, int index) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    return ((int64_t*)p->ptr)[index];
}

void cm_set_float(CPUMemory * p, int index, float value) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    ((float*)p->ptr)[index]=value;
}
void cm_set_double(CPUMemory * p, int index, double value) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    ((double*)p->ptr)[index]=value;
}
void cm_set_scomplex(CPUMemory * p, int index, scomplex value) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    ((scomplex*)p->ptr)[index]=value;
}
void cm_set_dcomplex(CPUMemory * p, int index, dcomplex value) {
    assert(p != NULL);
    assert(p->ptr != NULL);
    while(index < 0) index += p->n_elements;
    assert(index < p->n_elements);
    ((dcomplex*)p->ptr)[index]=value;
}



int sizeof_float() { return sizeof(float); }
int sizeof_double() { return sizeof(double); }
int sizeof_scomplex() { return sizeof(scomplex); }
int sizeof_dcomplex() { return sizeof(dcomplex); }
int sizeof_byte() { return sizeof(uint8_t); }
int sizeof_word() { return sizeof(uint16_t); }
int sizeof_dword() { return sizeof(uint32_t); }
int sizeof_qword() { return sizeof(uint64_t); }
int sizeof_int8() { return sizeof(int8_t); }
int sizeof_int16() { return sizeof(int16_t); }
int sizeof_int32() { return sizeof(uint32_t); }
int sizeof_int64() { return sizeof(uint64_t); }


BitVector * bv_new(int num_bits);
BitVector * bv_ptr(void * ptr, int size);
int         bv_get(BitVector * p, int index);
void        bv_set(BitVector * p, int index);
void        bv_flip(BitVector * p, int index);
BitVector* bv_encode_memory(CPUMemory * p);
CPUMemory* bv_decode_memory(BitVector * bits);
BitVector* bv_copy(BitVector * p);
BitVector* bv_not(BitVector * a);
BitVector* bv_op(BitVector * a, BitVector * b, BV_OP op);
void        bv_resize(BitVector * p, int bits);

BitVector* bv_encode_memory(CPUMemory * p) {
    BitVector* p = bv_ptr(p->ptr, p->n_elements * p->e_size * 8);
    assert(p != NULL);
    return p;
}
CPUMemory* bv_decode_memory(BitVector * bits) {
    CPUMemory * p = cm_new(bits->num_bits/sizeof(double), sizeof(double));
    assert(p != NULL);
    return p;
}

BitVector * bv_new(int num_bits) {
    BitVector * p = ((BitVector*))calloc(1,sizeof(BitVector));
    assert(p != NULL);
    p->bits = (uint8_t*)calloc(num_bits/8+1,sizeof(uint8_t));
    assert(p->bits != NULL);
    p->size = num_bits/8+1;
    p->num_bits = num_bits;
    return p;
}

BitVector * bv_ptr(void * ptr, int size) {
    BitVector * p = ((BitVector*))calloc(1,sizeof(BitVector));
    assert(p != NULL);
    int num_bits = size*sizeof(uint8_t);
    p->bits = (uint8_t*)calloc(num_bits/8+1,sizeof(uint8_t));
    memcpy(p->bits,ptr,size*sizeof(uint8_t));
    assert(p->bits != NULL);
    p->size = num_bits/8+1;
    p->num_bits = num_bits;
    return p;
}
int bv_get(BitVector * p, int index)
{
    assert(index < p->num_bits);    
    assert(p != NULL);
    assert(p->bits != NULL);
    assert(i < p->size);
    int i = index/sizeof(uint8_t);
    int l = index % 8;
    int bit = p->[i] & (1 << l);
    return bit;    
}

void bv_set(BitVector * p, int index) {
    assert(index < p->num_bits);    
    assert(p != NULL);
    assert(p->bits != NULL);
    assert(i < p->size);
    int i = index/sizeof(uint8_t);
    int l = index % 8;
    int mask = (1 << l);
    p->bits[i] |= mask;
}

void bv_flip(BitVector * p, int index) {
    assert(index < p->num_bits);    
    assert(p != NULL);
    assert(p->bits != NULL);
    assert(i < p->size);
    int i = index/sizeof(uint8_t);
    int l = index % 8;
    int mask = (1 << l);
    int bit = p->[i] & (1 << l);
    p->bits[i] &= ~bit;    
}

#define BV_ASSERT(x) { assert((x) != NULL); assert((x)->bits != NULL); }



BitVector* bv_copy(BitVector * p) {
    BitVector * r = bv_new(p->num_bits)
    assert(p != NULL);
    memcpy(r->bits,p->bits,p->size);
    r->num_bits = p->num_bits;
    r->size = p->size;
    return r;
}

BitVector* bv_op(BitVector * a, BitVector * b, BV_OP op) 
{
    uint8_t x,y,r;
    BV_ASSERT(a);
    BV_ASSERT(b);
    assert(a->num_bits == b->num_bits);
    BitVector * p = bv_copy(a);
    for(int i = 0; i < p->size; i++) {
        x = a->bits[i];
        y = b->bits[i];
        if(op == OP_NOR || op == OP_NAND || op == OP_NXOR) {
            x = ~x;
            y = ~y;
        }
        switch(op) {
            case OP_AND:
            case OP_NAND: r = x & y; break;
            case OP_OR:
            case OP_NOR: r = x | y; break;
            case OP_XOR:
            case OP_NXOR: r = x ^ y; break;
        }
        p->bits[i] = r;
    }
    return p;
}
BitVector* bv_not(BitVector * a) {
    BitVector * p = bv_copy(a);
    for(int i = 0; i < p->size; i++)
        p->bits[i] = ~a->bits[i];
    return p;
}
void  bv_resize(BitVector * p, int bits) {
    BV_ASSERT(p);
    uint8_t * n = (uint8_t*)calloc(bits/8+1,sizeof(uint8_t));
    assert(n != NULL);
    memcpy(n,p->bits,bits/8);
    free(p->bits);
    p->bits = n;
    p->num_bits = bits;
    p->size = bits/8+1;
}

enum SchemaType 
{
    SCHEMA_ZERO=0,
    SCHEMA_ONE=1,
    SCHEMA_DONTCARE=-1,
};

struct schema {
    std::vector<int8_t> vector;
};

struct bitvector 
{
    std::vector<bool> vector;

    bitvector(int num_bits) { vector.resize(num_bits*8); }
    bitvector(const bitvector & b) { vector = b.vector; }
    bitvector(void * ptr, int size) {
        encode(ptr,size);
    }
    void encode(void * ptr, int size) {
        vector.resize(size*8);
        uint8_t * p = ptr;
        for(size_t i = 0; i < size; i++)
        {       
            uint8_t x = p[i];     
            for(size_t j = 0; j < 8; j++)
            {
                bool val = x & (1 << j);
                vector[i*8 + j] = val;
            }
        }
    }
    std::vector<uint8_t> decode() {
        std::vector<uint8_t> d;
        d.resize(vector.size()/8 + 1);
        int c=0;
        for(size_t i = 0; i < vector.size(); i+=8)
        {
            uint8_t val = 0;
            for(size_t j = i; j < i+8; j++) {
                val |= vector[i] << j;
            }
            d[c++] = val;
        }
        return d;
    }
    bool operator[](size_t index) { return vector[index]; }
    bool __getitem(size_t index) { return vector[index]; }
    void __setitem(size_t index, bool v) { vector[index] = v; }
    void flip(size_t index) { vector[index] = !vector[index]; }
    bool get(size_t index) { return vector[index]; }
    void set(size_t index, bool val) { vector[index] = val;}


    bitvector& operator = (const bitvector& b) {
        vector = b.vector;
        return *this;
    }
    bitvector and(bitvector & b) { 
        bitvector r(*this);
        for(size_t i = 0; i < vector.size(); i++) 
            r[i] = vector[i] && b.vector[i];
    }
    bitvector or(bitvector & b) { 
        bitvector r(*this);
        for(size_t i = 0; i < vector.size(); i++) 
            r[i] = vector[i] || b.vector[i];
    }
    bitvector not() { 
        bitvector r(*this);
        for(size_t i = 0; i < vector.size(); i++) 
            r[i] = ! vector[i];
    }
};

template<typename T>
struct CpuMemory 
{
    CPUMemory * mem;

    CpuMemory(int size) {
        mem = cm_new(size, sizeof(T));
    }
    CpuMemory(int size, int element_size) {
        mem = cm_new(size,element_size);
    }
    ~CpuMemory() {
        if(mem) cm_free(mem);
    }

    T& operator[](size_t index) { return ((T*)mem->ptr)[index]; }
    T     __getitem(size_t index) { return ((T*)mem->ptr)[index]; }
    void  __setitem(size_t index, const T value) { ((T*)mem->ptr)[index] = value; }

    double   get_double(size_t i) { return cm_get_double(mem,i); }
    float    get_float(size_t i) { return cm_get_float(mem,i); }
    scomplex get_scomplex(size_t i) { return cm_get_scomplex(mem,i); }
    dcomplex get_dcomplex(size_t i) { return cm_get_dcomplex(mem,i); }
    uint8_t  get_byte(size_t i) { return cm_get_byte(mem,i); }
    uint16_t get_word(size_t i) { return cm_get_word(mem,i); }
    uint32_t get_dword(size_t i) { return cm_get_dword(mem,i); }
    uint64_t get_qword(size_t i) { return cm_get_qword(mem,i); }
    int8_t   get_int8(size_t i) { return cm_get_int8(mem,i); }
    int16_t  get_int16(size_t i) { return cm_get_int16(mem,i); }
    int32_t  get_int32(size_t i) { return cm_get_int32(mem,i); }
    int64_t  get_int64(size_t i) { return cm_get_int64(mem,i); }

    void    set_double(size_t i, double v) { cm_set_double(mem,i,v); }
    void    set_float(size_t i, float v) { cm_set_float(mem,i,v); }
    void    set_scomplex(size_t i, scomplex v) { cm_set_scomplex(mem,i,v); }
    void    set_dcomplex(size_t i, dcomplex v) { cm_set_dcomplex(mem,i,v); }
    void    set_byte(size_t i, uint8_t v) { cm_set_byte(mem,i,v); }
    void    set_word(size_t i, uint16_t v) { cm_set_word(mem,i,v); }
    void    set_dword(size_t i, uint32_t v) { cm_set_dword(mem,i,v); }
    void    set_qword(size_t i, uint64_t v) { cm_set_qword(mem,i,v); }
    void    set_int8(size_t i, int8_t v) { cm_set_int8(mem,i,v); }
    void    set_int16(size_t i, int16_t v) { cm_set_int16(mem,i,v); }
    void    set_int32(size_t i, int32_t v) { cm_set_int32(mem,i,v); }
    void    set_int64(size_t i, int64_t v) { cm_set_int64(mem,i,v); }

};

int main() 
{
    CpuMemory<float> cp(1024);
}