%module pffft
%{
#include "pffft.h"
%}
%include "std_vector.i"
%template(float_vector) std::vector<float>;
%include "pffft.h"
