// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "../fasst/STFTRepr.h"
#include "../fasst/Audio.h"
#include "gtest/gtest.h"

using namespace std;
using namespace Eigen;

TEST(TFRepr, allZeros) 
{
    // input: 8 samples, 1 channel, x=0, wlen=4
    // assert: every frames are zero
    fasst::Audio x(ArrayXXd::Zero(8, 1));
    int wlen = 4;

    fasst::STFTRepr STFT(x.samples(), wlen);
    ArrayMatrixXcd X = STFT.directFraming(x);
    ASSERT_EQ(X(0, 0).cols(), 1);
    ASSERT_EQ(STFT.bins(), 3);
    ASSERT_EQ(STFT.frames(), 4);
    for (int n = 0; n < STFT.frames(); n++)
    {
        for (int f = 0; f < STFT.bins(); f++)
        {
            ASSERT_EQ(X(f, n)(0).real(), 0.);
            ASSERT_EQ(X(f, n)(0).imag(), 0.);
        }
    }
}

TEST(TFRepr, allOnes) 
{
    // input: 8 samples, 1 channel, x=1, wlen=4
    // assert: every frames are equals except the edges
    fasst::Audio x(ArrayXXd::Ones(8, 1));
    int wlen = 4;

    fasst::STFTRepr STFT(x.samples(), wlen);
    ArrayMatrixXcd X = STFT.directFraming(x);
    for (int f = 0; f < STFT.bins(); f++)
    {
        ASSERT_EQ(X(f, 1)(0, 0).real(), X(f, 2)(0, 0).real());     //??
        ASSERT_EQ(X(f, 1)(0, 0).imag(), X(f, 2)(0, 0).imag());     //??
    }
}

TEST(TFRepr, firstSampleIsOne) 
{
    // input: 8 samples, 1 channel, (x=0, x(0)=1), wlen=4
    // assert: every frames are zero except the first one
    ArrayXXd array = ArrayXXd::Zero(8, 1);
    array(0, 0) = 1.;
    fasst::Audio x(array);
    int wlen = 4;

    fasst::STFTRepr STFT(x.samples(), wlen);
    ArrayMatrixXcd X = STFT.directFraming(x);
    for (int n = 1; n < STFT.frames(); n++)
    {
        for (int f = 0; f < STFT.bins(); f++)
        {
            ASSERT_EQ(X(f, n)(0, 0).real(), 0.);
            ASSERT_EQ(X(f, n)(0, 0).imag(), 0.);
        }
    }
}
