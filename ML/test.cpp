#include "viperfish.h"

Cublas _cublas;
Cublas *cublas = &_cublas;

void test1() 
{
    Vector a(10,{1,2,3,4,5,6,7,8,9,10});
    Vector b(10,{1,2,3,4,5,6,7,8,9,10});
    Vector r;
    r = a+b;
    r.print();
    r = a-b;
    r.print();
    r = a*b;
    r.print();
    float x = a.dot(b);
    std::cout << x << std::endl;
    x = a.nrm2();
    std::cout << x << std::endl;
    int idx = a.max_index();
    std::cout << idx << std::endl;
    a.download_host();
    std::cout << a[idx] << std::endl;
    idx = a.min_index();
    std::cout << a[idx] << std::endl;
}

int main(int argc, char *argv[]) 
{
    float x;
    Matrix m1(3,3);
    m1.fill(1);
    
}