%module pocketfft
%{
extern "C" {
#include "pocketfft.h"
}
%}
%include "std_vector.i"
%template(float_vector) std::vector<float>;

%include "pocketfft.h"
