%{
#include <Eigen/Eigen>
#include <iostream>
#include <vector>
%}

%inline %{
template<typename T> 
using RowVector = Eigen::Matrix<T,1,Eigen::Dynamic>;
template<typename T> 
using ColVector = Eigen::Matrix<T,Eigen::Dynamic,1>;
template<typename T>
using EigenArray = Eigen::Array<T,Eigen::Dynamic,1>;
template<typename T>
using Eigen2DArray = Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;


// a hack to index from lua
template<typename T, int X = Eigen::Dynamic, int Y = Eigen::Dynamic, int Z = Eigen::RowMajor>
class MatrixView 
{
private:
    Eigen::Matrix<T,X,Y,Z> *matrix;
    size_t row;
    size_t col;
    bool   bUseRows;
public:

    MatrixView(Eigen::Matrix<T,X,Y,Z> *m, size_t r)
    {
        matrix = m;
        row = r;
        col = 0;
        bUseRows = true;
    }

    T __getitem__(size_t i) { 
        if(bUseRows == true)
            return (*matrix)(row,i); 
        else 
            return (*matrix)(i,col); 
    }
    void  __setitem__(size_t i, const T& val) { 
        if(bUseRows == true)
            (*matrix)(row,i) = val; 
        else 
            (*matrix)(i,col) = val;
        
    }

    void use_rows() { bUseRows = true; }
    void use_cols() { bUseRows = false; }
    size_t rows() { return matrix->rows(); }
    size_t cols() { return matrix->cols(); }
    void   set_row(size_t r) { row = r; }
    void   set_col(size_t c) { col = c;}
    
};
%}

namespace Eigen
{    

    template<typename T, int X = Eigen::Dynamic, int Y = Eigen::Dynamic, int Z = Eigen::ColMajor>    
    class Matrix    
    {
    public:

        Matrix();
        Matrix(int x, int y);        
        Matrix(const Matrix<T,X,Y,Z>& m);

        T& operator()(size_t i, size_t j);
                
        void setRandom(size_t r, size_t c);
        void setIdentity(size_t r, size_t c);        
        void setZero(size_t r, size_t c);        
        void setOnes(size_t r, size_t c);
        
        Matrix<T,X,Y,Z> reshaped(size_t i, size_t j);
        Matrix<T,X,Y,Z> asDiagonal();

        T norm();
        T squaredNorm();

        bool all();
        bool allFinite();
        bool any();
        bool count();

        size_t rows() const;
        size_t cols() const;
        void resize(int x, int y);
        
        T* data();

        
        void normalize();
        Matrix<T,X,Y,Z> normalized();

        void fill(T v);
        Matrix<T,X,Y,Z> eval();
        bool hasNaN();
        size_t innerSize();
        size_t outerSize();
        bool isMuchSmallerThan(const Matrix<T,X,Y,Z>& n, T v);
        bool isOnes();
        bool isZero();

        // ugg        
        RowVector<T> row(size_t row);
        Eigen::Matrix<T,Eigen::Dynamic,1,Eigen::ColMajor> col(size_t col);

        Matrix<T,X,Y,Z> leftCols(size_t cols);
        Matrix<T,X,Y,Z> middleCols(size_t j, size_t cols);
        Matrix<T,X,Y,Z> rightCols(size_t cols);
        Matrix<T,X,Y,Z> topRows(size_t rows);
        Matrix<T,X,Y,Z> middleRows(size_t j, size_t rows);
        Matrix<T,X,Y,Z> bottomRows(size_t rows);
        Matrix<T,X,Y,Z> topLeftCorner(size_t i, size_t j);
        Matrix<T,X,Y,Z> topRightCorner(size_t i, size_t j);
        
        Matrix<T,X,Y,Z> adjoint();
        Matrix<T,X,Y,Z> transpose();
        Matrix<T,X,Y,Z> diagonal();
        Matrix<T,X,Y,Z> reverse();
        Matrix<T,X,Y,Z> replicate(size_t i, size_t j);
        
        void adjointInPlace();
        void transposeInPlace();
        void reverseInPlace();

        
        T sum();
        T prod();
        T mean() ;
        T minCoeff();
        T maxCoeff();
        T trace();  
        
        
        Matrix<T,X,Y,Z> cwiseAbs();
        Matrix<T,X,Y,Z> cwiseAbs2();
        Matrix<T,X,Y,Z> cwiseProduct(const Matrix<T,X,Y,Z>& q);
        Matrix<T,X,Y,Z> cwiseQuotient(const Matrix<T,X,Y,Z>& q);
        Matrix<T,X,Y,Z> cwiseInverse();
        Matrix<T,X,Y,Z> cwiseSqrt();
        Matrix<T,X,Y,Z> cwiseMax(Matrix<T,X,Y,Z>& q);
        Matrix<T,X,Y,Z> cwiseMin(Matrix<T,X,Y,Z>& q);
        //Matrix<T,X,Y,Z> cwiseEqual(Matrix<T,X,Y,Z>& q);
        //Matrix<T,X,Y,Z> cwiseNotEqual(Matrix<T,X,Y,Z>& q);

        bool operator == (const Matrix<T,X,Y,Z> &m);
        bool operator != (const Matrix<T,X,Y,Z> &m);        
        bool operator >= (const Matrix<T,X,Y,Z> &m);
        bool operator > (const Matrix<T,X,Y,Z> &m);
        //bool operator <= (const Matrix<T,X,Y,Z> &m);
        //bool operator < (const Matrix<T,X,Y,Z> &m);

        Matrix<T,X,Y,Z>& operator=(const Matrix<T,X,Y,Z> & m);

        Matrix<T,X,Y,Z>& operator *= (const Matrix<T,X,Y,Z> & m);
        Matrix<T,X,Y,Z>& operator += (const Matrix<T,X,Y,Z> & m);
        Matrix<T,X,Y,Z>& operator -= (const Matrix<T,X,Y,Z> & m);

        Matrix<T,X,Y,Z>& operator *= (const T f);
        Matrix<T,X,Y,Z>& operator /= (const T f);
        
        Matrix<T,X,Y,Z> operator * (const Matrix<T,X,Y,Z> & m);
        Matrix<T,X,Y,Z> operator + (const Matrix<T,X,Y,Z> & m);
        Matrix<T,X,Y,Z> operator - (const Matrix<T,X,Y,Z> & m);
        Matrix<T,X,Y,Z> operator - ();

        RowVector<T> operator * (const RowVector<T> & v);
        RowVector<T> operator + (const RowVector<T> & v);
        RowVector<T> operator - (const RowVector<T> & v);

        ColVector<T> operator * (const ColVector<T> & v);
        ColVector<T> operator + (const ColVector<T> & v);
        ColVector<T> operator - (const ColVector<T> & v);
        
        Matrix<T,X,Y,Z> operator * (const T f);
        Matrix<T,X,Y,Z> operator / (const T f);
        
        
        //Matrix<T,X,Y,Z> lazyAssign(const Matrix<T,X,Y,Z>& other);
        size_t nonZeros();

        
        %extend {
            MatrixView<T> __getitem__(size_t i) { return MatrixView<T>($self,i); }

            T lpNorm1() { return $self->lpNorm<1>(); }
            T lpNorm() { return $self->lpNorm<Eigen::Infinity>(); }
                        
            Eigen::Matrix<T,X,Y,Z> addToEachRow(RowVector<T> & v)    {
                Eigen::Matrix<T,X,Y,Z> r(*$self);        
                r = r.rowwise() + v;
                return r;
            }
            Eigen::Matrix<T,X,Y,Z> addToEachCol(ColVector<T> & v)    {
                Eigen::Matrix<T,X,Y,Z> r(*$self);        
                r = r.colwise() + v;
                return r;
            }
              
            RowVector<T> row_vector(size_t row) { RowVector<T> r; r = $self->row(row); return r; }    
            
            void set_row(size_t row, std::vector<T> & v)    {        
                for(size_t i = 0; i < $self->cols(); i++)
                    (*$self)(row,i) = v[i];
            }
            void set_row_vector(size_t row, RowVector<T> & v)    {        
                for(size_t i = 0; i < $self->cols(); i++)
                    (*$self)(row,i) = v(i);
            }
            void set_row_matrix(size_t row, Eigen::Matrix<T,X,Y,Z> & v, size_t src_row)    {        
                for(size_t i = 0; i < $self->cols(); i++)
                    (*$self)(row,i) = v(src_row,i);
            }
            
            ColVector<T> col_vector(size_t col) { ColVector<T> r; r = $self->col(col); return r; }
            
            void set_col(size_t col, std::vector<T> & v)    {
                assert($self->rows() == v.size());        
                for(size_t i = 0; i < $self->rows(); i++)
                    (*$self)(i,col) = v[i];
            }
            void set_col_vector(size_t col, ColVector<T> & v)    {
                assert($self->rows() == v.size());        
                for(size_t i = 0; i < $self->rows(); i++)
                    (*$self)(i,col) = v(i);
            }
            void set_col_matrix(size_t col, Eigen::Matrix<T,X,Y,Z> & v, size_t row)    {
                assert($self->rows() == v.cols());
                assert($self->cols() == v.rows());
                for(size_t i = 0; i < $self->rows(); i++)
                    (*$self)(i,col) = v(row,i);

            }
            
            void print()     {
                std::cout << *$self << std::endl;
            }
            
            Eigen::Matrix<T,X,Y,Z> t()     {
                return $self->transpose();
            }
            
            Eigen::Matrix<T,X,Y,Z> slice(int first_r,int first_c, int last_r=-1, int last_c=-1)    {
                if(last_r = -1) last_r = $self->rows()-1;
                if(last_c = -1) last_c = $self->cols()-1;
                auto r = (*$self)(Eigen::seq(first_r,last_r),Eigen::seq(first_c,last_c));
                Eigen::Matrix<T,X,Y,Z> ret(r.rows(),r.cols());
                for(size_t i = 0; i < r.rows(); i++)
                    for(size_t j = 0; j < r.cols(); j++)
                        ret(i,j) = r(i,j);
                return ret;
            }

            Eigen::Matrix<T,X,Y,Z> sliceN1(int first_r,int first_rn, int first_c, int last_c=-1)    {        
                if(last_c = -1) last_c = $self->cols()-1;
                auto r = (*$self)(Eigen::seqN(first_r,first_rn),Eigen::seq(first_c,last_c));
                Eigen::Matrix<T,X,Y,Z> ret(r.rows(),r.cols());
                for(size_t i = 0; i < r.rows(); i++)
                    for(size_t j = 0; j < r.cols(); j++)
                        ret(i,j) = r(i,j);
                return ret;
            }

            Eigen::Matrix<T,X,Y,Z> sliceN2(int first_r,int first_c, int first_cn, int last_r=-1)    {                
                auto r = (*$self)(Eigen::seq(first_r,last_r),Eigen::seqN(first_c,first_cn));
                Eigen::Matrix<T,X,Y,Z> ret(r.rows(),r.cols());
                for(size_t i = 0; i < r.rows(); i++)
                    for(size_t j = 0; j < r.cols(); j++)
                        ret(i,j) = r(i,j);
                return ret;
            }

            Eigen::Matrix<T,X,Y,Z> slicen(int first_r,int first_rn, int first_c, int first_cn)    {        
                auto r = (*$self)(Eigen::seqN(first_r,first_rn),Eigen::seqN(first_c,first_cn));
                Eigen::Matrix<T,X,Y,Z> ret(r.rows(),r.cols());
                for(size_t i = 0; i < r.rows(); i++)
                    for(size_t j = 0; j < r.cols(); j++)
                        ret(i,j) = r(i,j);
                return ret;
            }

            void rowwise_add(RowVector<T> & r) {
                $self->rowwise() += r;
            }
            void rowwise_sub(RowVector<T> & r) {
                $self->rowwise() -= r;
            }
            //void rowwise_mul(RowVector<T> & r) {
            //    $self->rowwise() *= r;
            //}
            //void rowwise_div(RowVector<T> & r) {
            //    $self->rowwise() /= r;
            //}

            void colwise_add(ColVector<T> & r) {
                $self->colwise() += r;
            }
            void colwise_sub(ColVector<T> & r) {
                $self->colwise() -= r;
            }
            //void colwise_mul(ColVector<T> & r) {
            //    $self->colwise() *= r;
            //}
            //void colwise_div(ColVector<T> & r) {
            //    $self->colwise() /= r;
            //}

            Eigen2DArray<T> array() const { return $self->array(); }            

            /*
            Matrix<T,X,Y,Z> adjointInto(Matrix<T,X,Y,Z> & m);
            Matrix<T,X,Y,Z> transposeInto(Matrix<T,X,Y,Z> & m);
            Matrix<T,X,Y,Z> diagonalInto(Matrix<T,X,Y,Z> & m);
            Matrix<T,X,Y,Z> reverseInto(Matrix<T,X,Y,Z> & m);
            Matrix<T,X,Y,Z> replicateInto(Matrix<T,X,Y,Z> & m,size_t i, size_t j);
            */
        }
        
    };
}


template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> abs(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().abs()(); return r;}
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> inverse(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().inverse()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> exp(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().exp()(); return r;  }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> log(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().log()(); return r;  }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> log1p(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().log1p()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> log10(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().log10()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> pow(const Matrix<T,X,Y,Z> & matrix,const T& b) { Matrix<T,X,Y,Z> r; r = matrix.array().pow(b)(); return r;}
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> sqrt(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().sqrt()(); return r;}
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> rsqrt(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().rsqrt()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> square(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().square()(); return r;}
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> cube(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().cube()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> abs2(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().abs2()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> sin(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().sin()(); return r;}
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> cos(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().cos()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> tan(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().tan()(); return r;}
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> asin(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().asin()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> acos(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().acos()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> atan(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().atan()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> sinh(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().sinh()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> cosh(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().cosh()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> tanh(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().tanh()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> ceil(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().ceil()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> floor(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().floor()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> round(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().round()(); return r; }


template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> asinh(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().asinh()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> acosh(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().acosh()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> atanh(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().atanh()(); return r; }
template<typename T, int X, int Y, int Z> Matrix<T,X,Y,Z> rint(const Matrix<T,X,Y,Z> & matrix) { Matrix<T,X,Y,Z> r; r = matrix.array().rint()(); return r; }


template<typename T, int X, int Y, int Z> 
void random(Matrix<T,X,Y,Z> & matrix, int x, int y) { matrix.random(x,y); }
template<typename T, int X, int Y, int Z> 
void random(Matrix<T,X,Y,Z> & matrix) { matrix.random(); }
template<typename T, int X, int Y, int Z> 
void identity(Matrix<T,X,Y,Z> & matrix,int x, int y) { matrix.identity(x,y); }
template<typename T, int X, int Y, int Z> 
void identity(Matrix<T,X,Y,Z> & matrix) { matrix.identity(); }
template<typename T, int X, int Y, int Z> 
void zero(Matrix<T,X,Y,Z> & matrix,int x, int y) { matrix.zero(x,y); }
template<typename T, int X, int Y, int Z> 
void zero(Matrix<T,X,Y,Z> & matrix) { matrix.zero(); }
template<typename T, int X, int Y, int Z> 
void ones(Matrix<T,X,Y,Z> & matrix,int x, int y) { matrix.ones(x,y); }
template<typename T, int X, int Y, int Z> 
void ones(Matrix<T,X,Y,Z> & matrix) { matrix.ones(); }

template<typename T, int X, int Y, int Z> 
T get(Matrix<T,X,Y,Z> & matrix,int i, int j) { return matrix.get(i,j); }

template<typename T, int X, int Y, int Z> 
void set(Matrix<T,X,Y,Z> & matrix,int i, int j, T v) { matrix.set(i,j,v); }

template<typename T, int X, int Y, int Z> 
T norm(Matrix<T,X,Y,Z> & matrix) { return matrix.norm(); }

template<typename T, int X, int Y, int Z> 
T squaredNorm(Matrix<T,X,Y,Z> & matrix) { return matrix.squaredNorm(); }

template<typename T, int X, int Y, int Z> 
bool all(Matrix<T,X,Y,Z> & matrix) { return matrix.all(); }

template<typename T, int X, int Y, int Z> 
bool allFinite(Matrix<T,X,Y,Z> & matrix) { return matrix.allFinite(); }

template<typename T, int X, int Y, int Z> 
bool any(Matrix<T,X,Y,Z> & matrix) { return matrix.any(); }

template<typename T, int X, int Y, int Z> 
bool count(Matrix<T,X,Y,Z> & matrix) { return matrix.count(); }

template<typename T, int X, int Y, int Z> 
size_t rows(Matrix<T,X,Y,Z> & matrix) { return matrix.rows(); }

template<typename T, int X, int Y, int Z> 
size_t cols(Matrix<T,X,Y,Z> & matrix) { return matrix.cols(); }

template<typename T, int X, int Y, int Z> 
void resize(Matrix<T,X,Y,Z> & matrix,int x, int y) { matrix.resize(x,y); }

template<typename T, int X, int Y, int Z> 
void normalize(Matrix<T,X,Y,Z> & matrix) { matrix.normalize(); }

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  normalized(Matrix<T,X,Y,Z> & matrix) { return matrix.normalized(); }    


template<typename T, int X, int Y, int Z> 
void fill(Matrix<T,X,Y,Z> & matrix,T v) { matrix.fill(v); }

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  eval(Matrix<T,X,Y,Z> & matrix) { return Matrix<T,X,Y,Z> (matrix.eval()); }

template<typename T, int X, int Y, int Z> 
bool hasNaN(Matrix<T,X,Y,Z> & matrix) { return matrix.hasNaN(); }

template<typename T, int X, int Y, int Z> 
size_t innerSize(Matrix<T,X,Y,Z> & matrix) { return matrix.innerSize(); }

template<typename T, int X, int Y, int Z> 
size_t outerSize(Matrix<T,X,Y,Z> & matrix) { return matrix.outerSize(); }    

template<typename T, int X, int Y, int Z> 
bool isMuchSmallerThan(Matrix<T,X,Y,Z> & matrix,const Matrix<T,X,Y,Z> & n, T v) { return matrix.isMuchSmallerThan(n,v); }

template<typename T, int X, int Y, int Z> 
bool isOnes(Matrix<T,X,Y,Z> & matrix) { return matrix.isOnes(); }

template<typename T, int X, int Y, int Z> 
bool isZero(Matrix<T,X,Y,Z> & matrix) { return matrix.isZero(); }

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  adjoint(Matrix<T,X,Y,Z> & matrix)  { return matrix.adjoint(); }

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  transpose(Matrix<T,X,Y,Z> & matrix) { return matrix.tranpose(); }

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  diagonal(Matrix<T,X,Y,Z> & matrix) { return matrix.diagonal(); }        

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  reverse(Matrix<T,X,Y,Z> & matrix) { return matrix.revese(); }    

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  replicate(Matrix<T,X,Y,Z> & matrix, size_t i, size_t j) { return matrix.replicate(i,j); }
    

template<typename T, int X, int Y, int Z> 
T sum(Matrix<T,X,Y,Z> & matrix)    {        
    return matrix.sum();        
}

template<typename T, int X, int Y, int Z> 
T prod(Matrix<T,X,Y,Z> & matrix)    {        
    return matrix.prod();        
}

template<typename T, int X, int Y, int Z> 
T mean(Matrix<T,X,Y,Z> & matrix)    {        
    return matrix.mean();        
}

template<typename T, int X, int Y, int Z> 
T minCoeff(Matrix<T,X,Y,Z> & matrix)    {        
    return matrix.minCoeff();        
}

template<typename T, int X, int Y, int Z> 
T maxCoeff(Matrix<T,X,Y,Z> & matrix)    {        
    return matrix.maxCoeff();        
}    

template<typename T, int X, int Y, int Z> 
T trace(Matrix<T,X,Y,Z> & matrix)    {        
    return matrix.trace();        
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  addToEachRow(Matrix<T,X,Y,Z>  & m, Matrix<T,X,Y,Z>  & v)    {
    Matrix<T,X,Y,Z>  r(m);        
    r = r.rowwise() + RowVector<T>(v);
    return r;
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  addToEachCol(Matrix<T,X,Y,Z>  & m, Matrix<T,X,Y,Z>  & v)    {
    Matrix<T,X,Y,Z>  r(m);        
    r = r.colwise() + ColVector<T>(v);
    return r;
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  cwiseAbs(Matrix<T,X,Y,Z>  & matrix)    {
    EigenMatrix<T,X,Y,Z>  r = matrix.cwiseAbs();
    return r;
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  cwiseAbs2(Matrix<T,X,Y,Z>  & matrix)    {
    EigenMatrix<T,X,Y,Z>  r = matrix.cwiseAbs2();
    return r;
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  cwiseProduct(Matrix<T,X,Y,Z>  & matrix,const Matrix<T,X,Y,Z> & q)    {
    EigenMatrix<T,X,Y,Z>  r = matrix.cwiseProduct(q); 
    return r;
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  cwiseQuotient(Matrix<T,X,Y,Z>  & matrix, const Matrix<T,X,Y,Z> & q)    {
    EigenMatrix<T,X,Y,Z>  r = matrix.cwiseQuotient(q); 
    return r;
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  cwiseInverse(Matrix<T,X,Y,Z>  & matrix)    {
    EigenMatrix<T,X,Y,Z>  r = matrix.cwiseInverse();
    return r;
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  cwiseSqrt(Matrix<T,X,Y,Z>  & matrix)    {
    EigenMatrix<T,X,Y,Z>  r = matrix.cwiseSqrt();
    return r;
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  cwiseMax(Matrix<T,X,Y,Z>  & matrix, Matrix<T,X,Y,Z> & q)    {
    EigenMatrix<T,X,Y,Z>  r = matrix.cwiseMin(q);
    return r;        
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  cwiseMin(Matrix<T,X,Y,Z>  & matrix, Matrix<T,X,Y,Z> & q)    {
    EigenMatrix<T,X,Y,Z>  r = matrix.cwiseMin(q);
    return r;
}


template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  slice(Matrix<T,X,Y,Z>  & matrix,int first_r,int first_c, int last_r=-1, int last_c=-1)    {
    return matrix.slice(first_r,first_c,last_r,last_c);
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  sliceN1(Matrix<T,X,Y,Z>  & matrix,int first_r,int first_rn, int first_c, int last_c=-1)    {        
    return matrix.sliceN1(first_r,first_rn,first_c,last_c);
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  sliceN2(Matrix<T,X,Y,Z>  & matrix,int first_r,int first_c, int first_cn, int last_r=-1)    {                
    return matrix.sliceN2(first_r, first_c, first_cn, last_r);
}

template<typename T, int X, int Y, int Z> 
Matrix<T,X,Y,Z>  slicen(Matrix<T,X,Y,Z>  & matrix,int first_r,int first_rn, int first_c, int first_cn)    {        
    return matrix.slicen(first_r,first_rn,first_c,first_cn);
}

template<typename T, int X, int Y, int Z> 
Eigen2DArray <T> array(Matrix<T,X,Y,Z>  & matrix) { return matrix.array(); }
