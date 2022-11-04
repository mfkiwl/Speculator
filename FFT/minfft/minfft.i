%module minfft
%{
#define complex
#define MINFFT_SINGLE
#include <minfft.h>
#include <vector>
#include <complex>
%}
%include "std_vector.i"
%include "std_complex.i"

#define complex
#define MINFFT_SINGLE
%include "minfft.h"


%template(float_vector)     std::vector<float>;
