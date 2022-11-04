%{
#include <Eigen/Eigen>
#include <iostream>
#include <vector>
%}


template<typename T> 
using RowVector = Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>;

template<typename T> 
using ColVector = Eigen::Matrix<T,Eigen::Dynamic,1,Eigen::ColMajor>;



namespace Eigen
{
    
    template<class T, int X=Dynamic, int Y=1, int Z=ColMajor>
    class Matrix
    {    
    public:
        
        Matrix();
        Matrix(size_t i);        
        Matrix(const Matrix& v);
                    
        void setRandom(size_t size);
        void setZero(size_t size);
        void setOnes(size_t size);

        size_t  innerStride();
        T*      data();
        void    resize(size_t size);
        void    resizeLike(const Matrix<T,X,Y,Z> & v);
        void    conservativeResize(size_t size);

        T coeff(size_t i);
        T& coeffRef(size_t i);
        size_t cols();
        T dot(const Matrix<T,X,Y,Z> & b);
        size_t size() const;
        void fill(T v);
        T norm();
        T squaredNorm();
        Matrix<T,X,Y,Z> eval();
        void normalize();
        Matrix<T,X,Y,Z> normalized();

                
        T operator()(size_t i);
        T operator[](size_t i);
  
        Matrix<T,X,Y,Z>& operator += (const Matrix<T,X,Y,Z>& b);        
        Matrix<T,X,Y,Z>& operator -= (const Matrix<T,X,Y,Z>& b);
        Matrix<T,X,Y,Z> operator + (const Matrix<T,X,Y,Z>& b);        
        Matrix<T,X,Y,Z> operator - (const Matrix<T,X,Y,Z>& b);
            
        Matrix<T,X,Y,Z> cwiseAbs();
        Matrix<T,X,Y,Z> cwiseInverse();
        Matrix<T,X,Y,Z> cwiseSqrt();
        Matrix<T,X,Y,Z> cwiseAbs2();

        Matrix<T,Eigen::Dynamic,1,Eigen::ColMajor> transpose();

        %extend {
            void print() {        std::cout << *$self << std::endl;    }
        
            Matrix<T,Eigen::Dynamic,1,Eigen::ColMajor> t() { return $self->transpose().eval();}    
            
            T     __getitem__(int i) { while(i < 0) i += $self->rows(); return (*$self)[i]; }
            void  __setitem__(int i, T val) { while(i < 0) i += $self->rows(); (*$self)[i] = val; }

            T operator * (const Matrix<T,X,Y,Z>& b) 
            { 
                return $self->dot(b);        
            }
            
            Eigen::Matrix<T,X,Y,Z> operator * (const T b) 
            { 
                Eigen::Matrix<T,X,Y,Z> r(*$self);
                r = r * b;
                return r;
            }

            Eigen::Matrix<T,X,Y,Z> operator / (const T b)  
            { 
                Eigen::Matrix<T,X,Y,Z> r(*$self);
                r = r / b;
                return r;
            }

            Eigen::Matrix<T,X,Y,Z>& operator += (const T b) 
            { 
                *$self += b;
                return *$self;
            }
            Eigen::Matrix<T,X,Y,Z>& operator -= (const T b) 
            { 
                *$self -= b;
                return *$self;
            }    
            Eigen::Matrix<T,X,Y,Z>& operator *= (const T b) 
            { 
                *$self *= b;
                return *$self;
            }

            T lpNorm1() { return $self->lpNorm<1>(); }
            T lpNorm()  { return $self->lpNorm<Eigen::Infinity>(); }    

            Eigen::Matrix<T,X,Y,Z> LinSpaced(T num, T a, T b)
            {        
                *$self = Eigen::Matrix<T,X,Y,Z>::LinSpaced(num,a,b);
                return *$self;
            }
        }
    };
}

template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  abs(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.abs(); return r;}
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  inverse(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.inverse(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  exp(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().exp(); return r;  }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  log(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().log(); return r;  }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  log1p(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().log1p(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  log10(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().log10(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  pow(Matrix<T,X,Y,Z> & Matrix ,const T& b) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().pow(b); return r;}
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  sqrt(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.sqrt(); return r;}
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  rsqrt(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().rsqrt(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  square(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().square(); return r;}
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  cube(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().cube(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  abs2(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.abs2(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  sin(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().sin(); return r;}
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  cos(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().cos(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  tan(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().tan(); return r;}
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  asin(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().asin(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  acos(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().acos(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  atan(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().atan(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  sinh(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().sinh(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  cosh(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().cosh(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  tanh(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().tanh(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  ceil(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().ceil(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  floor(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().floor(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z>  round(Matrix<T,X,Y,Z> & Matrix ) { Matrix<T,X,Y,Z>  r; r = Matrix.Matrix.array().round(); return r; }

template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> asinh(const Matrix<T,X,Y,Z> & Matrix) { return Matrix<T,X,Y,Z>(Matrix.Matrix.array().asinh().matrix()); }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> acosh(const Matrix<T,X,Y,Z> & Matrix) { return Matrix<T,X,Y,Z>(Matrix.Matrix.array().acosh().matrix()); }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> atanh(const Matrix<T,X,Y,Z> & Matrix) { return Matrix<T,X,Y,Z>(Matrix.Matrix.array().atanh().matrix()); }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> rint(const Matrix<T,X,Y,Z> & Matrix)  { return Matrix<T,X,Y,Z>(Matrix.Matrix.array().rint().matrix());  }

template<typename T, int X, int Y, int Z>
void random(Matrix<T,X,Y,Z> & v, int i) { v.setRandom(i); }

template<typename T, int X, int Y, int Z>
void random(Matrix<T,X,Y,Z> & Matrix) { Matrix.setRandom(Matrix.size()); }

template<typename T, int X, int Y, int Z>
void zero(Matrix<T,X,Y,Z> & Matrix) { Matrix.setZero(Matrix.size()); }    

template<typename T, int X, int Y, int Z>
void ones(Matrix<T,X,Y,Z> & Matrix) { Matrix.setOnes(Matrix.size()); }

template<typename T, int X, int Y, int Z>
size_t  innerStride(Matrix<T,X,Y,Z> & Matrix) { return Matrix.innerStride(); }

template<typename T, int X, int Y, int Z>
void    resize(Matrix<T,X,Y,Z> & Matrix, size_t size) { Matrix.resize(size); }

template<typename T, int X, int Y, int Z>
void    resizeLike(Matrix<T,X,Y,Z> & Matrix, const Matrix<T,X,Y,Z> & v) { Matrix.resizeLike(v.Matrix); }

template<typename T, int X, int Y, int Z>
void    conservativeResize(Matrix<T,X,Y,Z> & Matrix,size_t size) { Matrix.conservativeResize(size); }

template<typename T, int X, int Y, int Z>
T coeff(Matrix<T,X,Y,Z> & Matrix,size_t i) { return Matrix.coeff(i); }

template<typename T, int X, int Y, int Z>
T& coeffRef(Matrix<T,X,Y,Z> & Matrix, size_t i) { return Matrix.coeffRef(i); }

template<typename T, int X, int Y, int Z>
void print(Matrix<T,X,Y,Z> & Matrix)     {    Matrix.print();  }

template<typename T, int X, int Y, int Z>
size_t cols(Matrix<T,X,Y,Z> & Matrix) { return Matrix.cols(); }

template<typename T, int X, int Y, int Z>
T dot(Matrix<T,X,Y,Z> & Matrix, const Matrix<T,X,Y,Z> & b) { return Matrix.dot(b.Matrix);  }        

template<typename T, int X, int Y, int Z>
size_t size(Matrix<T,X,Y,Z> & Matrix) { return Matrix.size(); }

template<typename T, int X, int Y, int Z>
void fill(Matrix<T,X,Y,Z> & Matrix,T v) { Matrix.fill(v); }

template<typename T, int X, int Y, int Z>
Matrix<T,X,Y,Z> fill(size_t i, T v) { Matrix<T,X,Y,Z> Matrix(i); Matrix.fill(v); return Matrix; }

template<typename T, int X, int Y, int Z>
T norm(Matrix<T,X,Y,Z> & Matrix) { return Matrix.norm(); }

template<typename T, int X, int Y, int Z>
T squaredNorm(Matrix<T,X,Y,Z> & Matrix) { return Matrix.squaredNorm(); }

template<typename T, int X, int Y, int Z>
Matrix<T,X,Y,Z>& eval(Matrix<T,X,Y,Z> & Matrix) { return Matrix.eval(); }

template<typename T, int X, int Y, int Z>
void normalize(Matrix<T,X,Y,Z> & Matrix) { Matrix.normalize(); }

template<typename T, int X, int Y, int Z>
Matrix<T,X,Y,Z> normalized(Matrix<T,X,Y,Z> & Matrix) { return Matrix.normalized();  }    

//template<typename T, int X, int Y, int Z>
//Matrix<T,X,Y,Z> transpose(Matrix<T,X,Y,Z> & Matrix) { return Matrix<T,X,Y,Z>(Matrix.transpose().eval());}    
template<typename T, int X, int Y, int Z>
Matrix<T,X,Y,Z> cwiseAbs(Matrix<T,X,Y,Z> & Matrix) {  return Matrix<T,X,Y,Z>(Matrix.cwiseAbs());  }
template<typename T, int X, int Y, int Z>
Matrix<T,X,Y,Z> cwiseInverse(Matrix<T,X,Y,Z> & Matrix) { return Matrix<T,X,Y,Z>(Matrix.cwiseInverse()); }    
template<typename T, int X, int Y, int Z>
Matrix<T,X,Y,Z> cwiseSqrt(Matrix<T,X,Y,Z> & Matrix) { return Matrix<T,X,Y,Z>(Matrix.cwiseSqrt()); }
template<typename T, int X, int Y, int Z>
Matrix<T,X,Y,Z> cwiseAbs2(Matrix<T,X,Y,Z> & Matrix) { return Matrix<T,X,Y,Z>(Matrix.cwiseAbs2()); }
