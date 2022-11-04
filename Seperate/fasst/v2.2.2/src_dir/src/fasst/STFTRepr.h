// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_TFREPR_H
#define FASST_TFREPR_H

#include "typedefs.h"
#include "Audio.h"
#include "Abstract_TFRepr.h"

namespace fasst {
class Audio;
class Source;
class Sources;
/*!
 This class allows the computation of the time-frequency representation of some audio signal using the STFT representation.
 This class derives Abstract_TFRepr.
*/
class STFTRepr : public Abstract_TFRepr {
    public:
        // ------------ Constructor ------------
        /*!
        The constructor of the class for STFT direct and inverse transforms
        \param nOrigSamples number of time samples
        \param wlen the window length _ie._ the length (in audio samples) of one time frame
        */
		STFTRepr(const int nOrigSamples, const int wlen);

        // ------------ Methods ------------
        /*!
        This method computes the STFT direct transform of an audio signal 
        (including framing as STFT is a framed representation by default).
        The output has the following size :
        \param x Audio signal
        \return STFT of input signal : (F x N) x (1 x I) <=> (nFrames x nBins) x (1 x nChannels)
        */
        ArrayMatrixXcd directFraming(const Audio & x) const override;

        /*!
         This method computes the inverse STFT of input data.
         \param X STFT transform of an audio signal : (N x F) x (1 x I) <=> (nFrames x nBins) x (1 x nChannels)
         \return the signal in the time domain : (m_origSamples x I)  <=> (m_origSamples x nChannels)
         */
        ArrayXXd inverse(const ArrayMatrixXcd & X) const override;

        /*!
         This method computes the STFT, then applies the Wiener filter, then computes the inverse STFT and write the
		 filtered audio source on the disk.
         \param x the mixture audio signal
		 \param wiener A pointer to an object which implements Wiener_Interface
		 \dirname The directory where the filtered sources are saved
         \return the audio signal of each source
         */
		void Filter(const Audio & x, Wiener_Interface * wiener, const std::string & dirName) override ;

    private:

    };
}

#endif
