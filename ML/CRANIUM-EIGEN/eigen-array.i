%{
#include "Eigen"
#include <iostream>

%}

%include "std_complex.i"
%include "std_vector.i"

namespace Eigen 
{    
    template<class T, int _Rows=Dynamic, int _Cols=1>
    class Array
    {
    public:

        Array();        
        Array(const Array & a);
        Array(size_t size);
        //Array(size_t size, const T & val);
        ~Array();
        
        Array<T,_Rows,_Cols>& operator = (const Array & a);
        
        T& operator()(size_t i);
        T& operator[](size_t i);

        size_t size() const;
        void   resize(size_t size);
        
        void    setRandom();
        void    fill(T v);
        size_t  cols() const;
            
        
        %extend {            
            // lua is 1 index like fortran
            void    __setitem__(size_t i, T v) { (*$self)[i] = v; }
            T       __getitem__(size_t i) { return (*$self)[i]; }

            void println()
            {
                std::cout << *$self << std::endl;
            }
        }

            
        Array<T,_Rows,_Cols> operator + (const Array& b);        
        Array<T,_Rows,_Cols> operator - (const Array& b);
        Array<T,_Rows,_Cols> operator * (const Array& b);
        Array<T,_Rows,_Cols> operator / (const Array& b); 
         
    };
}

%inline %{
namespace Ops {
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> abs(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.abs(); }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> inverse(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.inverse();   }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> exp(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.exp();   }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> log(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.log();   }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> log1p(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.log1p();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> log10(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.log10();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> pow(const Eigen::Array<T,Eigen::Dynamic,1> & array,const T& b) { return array.pow(b);  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> sqrt(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.sqrt();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> rsqrt(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.rsqrt();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> square(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.square();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> cube(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.cube();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> abs2(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.abs2();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> sin(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.sin();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> cos(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.cos();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> tan(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.tan(); }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> asin(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.asin();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> acos(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.acos();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> atan(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.atan();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> sinh(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.sinh();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> cosh(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.cosh();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> tanh(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.tanh();  }    
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> ceil(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.ceil();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> floor(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.floor();  }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> round(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.round();  }


    template<typename T> Eigen::Array<T,Eigen::Dynamic,1> asinh(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.asinh(); }
    template<typename T> Eigen::Array<T,Eigen::Dynamic,1> acosh(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.acosh(); }
    template<typename T> Eigen::Array<T,Eigen::Dynamic,1> atanh(const Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.atanh(); }
    template<typename T> Eigen::Array<T,Eigen::Dynamic,1> rint(const Eigen::Array<T,Eigen::Dynamic,1> & array)  { return array.rint();  }

    template<typename T>
    size_t size(Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.size(); }  
    template<typename T>
    void    random(int i, Eigen::Array<T,Eigen::Dynamic,1> & array) { array.resize(i); array.setRandom(i); }
    template<typename T>
    void    random(Eigen::Array<T,Eigen::Dynamic,1> & array) { array.setRandom(); }    
    template<typename T>
    void    fill(Eigen::Array<T,Eigen::Dynamic,1> & array, T v) { array.fill(v); }
    template<typename T>
    size_t  cols(Eigen::Array<T,Eigen::Dynamic,1> & array) { return array.cols(); }

    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> operator + (T a, Eigen::Array<T,Eigen::Dynamic,1> & b) {
        Eigen::Array<T,Eigen::Dynamic,1> r(b);
        r = a + b;
        return r;
    }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> operator - (T a, Eigen::Array<T,Eigen::Dynamic,1> & b) {
        Eigen::Array<T,Eigen::Dynamic,1> r(b);
        r = a - b;
        return r;
    }
    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> operator * (T a, Eigen::Array<T,Eigen::Dynamic,1> & b) {
        Eigen::Array<T,Eigen::Dynamic,1> r(b);
        r = a * b;
        return r;
    }

    template<typename T>
    Eigen::Array<T,Eigen::Dynamic,1> scalar_array(size_t n, T v) {
        Eigen::Array<T,Eigen::Dynamic,1> r(n);
        r.fill(v);
        return r;
    }

    template<typename T>
    void resize(Eigen::Array<T,Eigen::Dynamic,1> & a, size_t n) { a.resize(n); }    
}
%}

