%module array
%{
#include "Eigen"
#include <iostream>
using namespace Eigen;
%}

%include "std_complex.i"
%include "std_vector.i"
%include "eigen-array.i"

%template(fcomplex) std::complex<float>;
%template(zcomplex) std::complex<double>;

%template(ArrayXf) Eigen::Array<float,Eigen::Dynamic,1>;
%template(ArrayXd) Eigen::Array<double,Eigen::Dynamic,1>;
%template(ArrayXi) Eigen::Array<int,Eigen::Dynamic,1>;
%template(ArrayXl) Eigen::Array<long,Eigen::Dynamic,1>;
%template(ArrayXc) Eigen::Array<std::complex<float>,Eigen::Dynamic,1>;
%template(ArrayXz) Eigen::Array<std::complex<double>,Eigen::Dynamic,1>;

%template(absf) Ops::abs<float>;
%template(abs2f) Ops::abs2<float>;
%template(inversef) Ops::inverse<float>;
%template(expf) Ops::exp<float>;
%template(logf) Ops::log<float>;
%template(log1pf) Ops::log1p<float>;
%template(log10f) Ops::log10<float>;
%template(powf) Ops::pow<float>;
%template(sqrtf) Ops::sqrt<float>;
%template(rsqrtf) Ops::rsqrt<float>;
%template(square) Ops::square<float>;
%template(cube) Ops::cube<float>;
%template(sinf) Ops::sin<float>;
%template(cosf) Ops::cos<float>;
%template(tanf) Ops::tan<float>;
%template(asinf) Ops::asin<float>;
%template(acosf) Ops::acos<float>;
%template(atanf) Ops::atan<float>;
%template(sinhf) Ops::sinh<float>;
%template(coshf) Ops::cosh<float>;
%template(tanhf) Ops::tanh<float>;
%template(asinhf) Ops::asinh<float>;
%template(acoshf) Ops::acosh<float>;
%template(atanhf) Ops::atanh<float>;
%template(floorf) Ops::floor<float>;
%template(ceilf) Ops::ceil<float>;
%template(roundf) Ops::round<float>;
%template(rintf) Ops::rint<float>;
%template(sizef) Ops::size<float>;
%template(randomf) Ops::random<float>;
%template(fillf) Ops::fill<float>;
%template(colsf) Ops::cols<float>;
%template(resizef) Ops::resize<float>;


%template(cfreal) std::real<float>;
%template(czreal) std::real<double>;

%template(cfimag) std::imag<float>;
%template(czimag) std::imag<double>;

%template(cfabs) std::abs<float>;
%template(czabs) std::abs<double>;

%template(cfarg) std::arg<float>;
%template(czarg) std::arg<double>;

%template(cfnorm) std::norm<float>;
%template(cznorm) std::norm<double>;

%template(cfproj) std::proj<float>;
%template(czproj) std::proj<double>;

%template(cfpolar) std::polar<float>;
%template(czpolar) std::polar<double>;

%template(cfexp) std::exp<float>;
%template(czexp) std::exp<double>;

%template(cflog) std::log<float>;
%template(czlog) std::log<double>;

%template(cflog10) std::log10<float>;
%template(czlog10) std::log10<double>;

%template(cfpow) std::pow<float>;
%template(czpow) std::pow<double>;

%template(cfsqrt) std::sqrt<float>;
%template(czsqrt) std::sqrt<double>;

%template(cfsin) std::sin<float>;
%template(czsin) std::sin<double>;
    
%template(cfcos) std::cos<float>;
%template(czcos) std::cos<double>;

%template(cftan) std::tan<float>;
%template(cztan) std::tan<double>;

%template(cfasin) std::asin<float>;
%template(czasin) std::asin<double>;
    
%template(cfacos) std::acos<float>;
%template(czacos) std::acos<double>;

%template(cfatan) std::atan<float>;
%template(czatan) std::atan<double>;

%template(cfsinh) std::sinh<float>;
%template(czsinh) std::sinh<double>;
    
%template(cfcosh) std::cosh<float>;
%template(czcosh) std::cosh<double>;

%template(cftanh) std::tanh<float>;
%template(cztanh) std::tanh<double>;

%template(cfasinh) std::asinh<float>;
%template(czasinh) std::asinh<double>;
    
%template(cfacosh) std::acosh<float>;
%template(czacosh) std::acosh<double>;

%template(cfatanh) std::atanh<float>;
%template(czatanh) std::atanh<double>;

    