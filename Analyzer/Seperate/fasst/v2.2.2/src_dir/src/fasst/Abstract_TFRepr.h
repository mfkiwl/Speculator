// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_ABSTRACT_TFREPR_H
#define FASST_ABSTRACT_TFREPR_H

#include "typedefs.h"
#include "Audio.h"
#include "Wiener_Interface.h"
#include <string>

using namespace std;
using namespace Eigen;

/*!
This class represents an base class for time-frequency representations.
It provides a default constructor to construct private members
*/
namespace fasst {
	class Abstract_TFRepr {
	public:
		/*!
		The default constructor
		\param nOrigSamples number of time samples
		\param wlen the window length _ie._ the length (in audio samples) of one time frame
		*/
		Abstract_TFRepr(const int nOrigSamples, const int wlen)
			: m_origSamples(nOrigSamples), m_wlen(wlen), m_samples(0), m_frames(0), m_bins(0)
		{};

		/*!
		This virtual method is provided to compute the direct framed time - frequency transform.
		The output has the following size :
		\param x Audio signal
		\return The transformed signal - output size must be choosen in sub classes
		*/
		virtual ArrayMatrixXcd directFraming(const Audio & x) const = 0;

		/*!
		This virtual method is provided to compute the inverse framed time - frequency transform.
		The output has the following size :
		\param X The transformed signal in frequency domain
		\return The signal in the time domain
		*/
		virtual ArrayXXd inverse(const ArrayMatrixXcd & X) const = 0;

		/*!
		This virtual method is provided to filter the input signal by using a Wiener Filter approach.
		\param x The input audio to be filtered
		\param wiener A pointer to an object which implements Wiener_Interface
		\param dirName The output directory where the filtered signals could be saved
		\return The signal in the temporal domain
		*/
		virtual void Filter(const Audio & x, Wiener_Interface * wiener, const std::string & dirName) = 0;

		virtual ~Abstract_TFRepr() {};
		/*!
		This method return the number of frequency bins
		\return Number of frequency bins
		*/
		inline int bins() const { return m_bins; };

		/*!
		This method return the number of time frames
		*/
		inline int frames() const { return m_frames; };

	protected:
		//  ----- Members (initialized by the constructor) -----
		int m_origSamples;      // original number of time samples
		int m_wlen;             // window / frame length	

		int m_samples;          // original number of time samples + zero padded samples
		int m_frames;           // number of frames
		int m_bins;             // number of frequency bins / number of reliable filters

	};
}

#endif