%module samples
%{
#include "sample.hpp"
#include "sample_dsp.hpp"

%}

%include "stdint.i"
%include "std_vector.i"
%include "std_math.i"

// Get the STL typemaps
%include "stl.i"

// Handle standard exceptions
%include "exception.i"
%exception
{
  try
  {
    $action
  }
  catch (const std::invalid_argument& e)
  {
    SWIG_exception(SWIG_ValueError, e.what());
  }  
}

%include "sample.hpp"
%include "sample_dsp.hpp"

%extend sample_vector<T>
{
    sample_vector<T> __add__(sample_vector<T> & b) {
        return (*$self) + b;
    }
    sample_vector<T> __sub__(sample_vector<T> & b) {
        return (*$self) - b;
    }
    sample_vector<T> __mul__(sample_vector<T> & b) {
        return (*$self) + b;
    }
};

%inline %{
    
    struct float_vector : public sample_vector<float>
    {
        float_vector() = default;
        float_vector(size_t n,size_t ch=1) : sample_vector<float>(n,ch) {

        }
        float __getitem__(size_t i) { return (*this)[i-1]; }
        void  __setitem__(size_t i, float x) { (*this)[i-1] = x; }

        float_vector __add__(float_vector& b) {
            float_vector r(size());
            for(size_t i = 0; i < size(); i++)
                r[i] = (*this)[i] + b[i];
            return r;
        }
        float_vector __sub__(float_vector& b) {
            float_vector r(size());
            for(size_t i = 0; i < size(); i++)
                r[i] = (*this)[i] - b[i];
            return r;
        }
        float_vector __mul__(float_vector& b) {
            float_vector r(size());
            for(size_t i = 0; i < size(); i++)
                r[i] = (*this)[i] * b[i];
            return r;                
        }   
    };

    struct complex_float_vector : public complex_vector<float>
    {
        complex_float_vector() = default;
        complex_float_vector(size_t n,size_t ch=1) : complex_vector<float>(n,ch) {

        }
        std::complex<float> __getitem__(size_t i) { return (*this)[i-1]; }
        void  __setitem__(size_t i, const std::complex<float> & x) { (*this)[i-1] = x; }

        complex_float_vector __add__(complex_float_vector& b) {
            complex_float_vector r(size());
            for(size_t i = 0; i < size(); i++)
                r[i] = (*this)[i] + b[i];
            return r;
        }
        complex_float_vector __sub__(complex_float_vector& b) {
            complex_float_vector r(size());
            for(size_t i = 0; i < size(); i++)
                r[i] = (*this)[i] - b[i];
            return r;
        }
        complex_float_vector __mul__(complex_float_vector& b) {
            complex_float_vector r(size());
            for(size_t i = 0; i < size(); i++)
                r[i] = (*this)[i] * b[i];
            return r;                
        }   
    };
%};


/* arrg swig won't interface sample_vector template
    std_vector.i should take allocator but doesnt right now
%template(float_vector) sample_vector<float>;
%template(double_vector) sample_vector<double>;
%template(float_complex_vector) complex_vector<float>;
%template(double_complex_vector) complex_vector<double>;
%template(int8_vector) sample_vector<signed char>;
%template(uint8_vector) sample_vector<unsigned char>;
%template(int16_vector) sample_vector<signed short>;
%template(uint16_vector) sample_vector<unsigned short>;
%template(int32_vector) sample_vector<signed int>;
%template(uint32_vector) sample_vector<unsigned int>;
%template(int64_vector) sample_vector<signed long>;
%template(uint64_vector) sample_vector<unsigned long>;
*/

%template(get_left_channel_float) get_left_channel<float>;
%template(get_right_channel_float) get_right_channel<float>;
%template(get_channel_float) get_channel<float>;

%template(interleave_float) interleave<float>;
%template(deinterleave_float) interleave<float>;
%template(copy_vector_float) copy_vector<float>;
%template(slice_vector_float) slice_vector<float>;
%template(copy_buffer_float) copy_buffer<float>;
%template(slice_buffer_float) slice_buffer<float>;
%template(stereo_split_float) split_stereo<float>;
%template(insert_front_float) insert_front<float>;

%template(containsOnlyZeros_float) containsOnlyZeros<float>;
%template(isAllPositiveOrZero_float) isAllPositiveOrZero<float>;
%template(isAllNegativeOrZero_float) isAllNegativeOrZero<float>;
%template(contains_float) contains<float>;
%template(max_float) max<float>;
%template(min_float) min<float>;
%template(maxIndex_float) maxIndex<float>;
%template(minIndex_float) minIndex<float>;
%template(printVector_float) printVector<float>;
%template(getFirstElement_float) getFirstElement<float>;
%template(getLastElement_float) getLastElement<float>;
%template(getEvenElements_float) getEvenElements<float>;
%template(getOddElements_float) getOddElements<float>;
%template(getEveryNthElementStartingFromK_float) getEveryNthElementStartingFromK<float>;
%template(fillVectorWith_float) fillVectorWith<float>;
%template(countOccurrencesOf_float) countOccurrencesOf<float>;
%template(sum_float) sum<float>;
%template(product_float) product<float>;
%template(mean_float) mean<float>;
%template(median_float) median<float>;
%template(variance_float) variance<float>;
%template(standardDeviation_float) standardDeviation<float>;
%template(norm1_float) norm1<float>;
%template(norm2_float) norm2<float>;
%template(normP_float) normP<float>;
%template(magnitude_float) magnitude<float>;
%template(multiplyInPlace_float) multiplyInPlace<float>;
%template(divideInPlace_float) divideInPlace<float>;
%template(addInPlace_float) addInPlace<float>;
%template(subtractInPlace_float) subtractInPlace<float>;
%template(absInPlace_float) absInPlace<float>;
%template(squareInPlace_float) squareInPlace<float>;
%template(squareRootInPlace_float) squareRootInPlace<float>;
%template(sort_float) sort<float>;
%template(reverse_float) reverse<float>;
%template(multiply_float) multiply<float>;
%template(divide_float) divide<float>;
%template(add_float) add<float>;
%template(subtract_float) subtract<float>;
%template(abs_float) abs<float>;
%template(square_float) square<float>;
%template(squareRoot_float) squareRoot<float>;
%template(scale_float) scale<float>;
%template(difference_float) difference<float>;
%template(zeros_float) zeros<float>;
%template(ones_float) ones<float>;
%template(range_float) range<float>;
%template(dotProduct_float) dotProduct<float>;
%template(euclideanDistance_float) euclideanDistance<float>;
%template(cosineSimilarity_float) cosineSimilarity<float>;
%template(cosineDistance_float) cosineDistance<float>;

%template(linearf) linear_interpolate<float>;
%template(cubicf) cubic_interpolate<float>;
%template(hermite1f) hermite1<float>;
%template(hermite2f) hermite2<float>;
%template(hermite3f) hermite3<float>;
%template(hermite4f) hermite4<float>;
%template(mixf) mix<float>;
%template(interp2xf) interp2x<float>;
%template(interp4xf) interp4x<float>;