// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_ERBLETREPR_H
#define FASST_ERBLETREPR_H

#include "typedefs.h"
#include "Sources.h"
#include "Audio.h"

/*!
This class allows the computation of the time-frequency representation of some audio signal using the ERBLET representation.
This class derives Abstract_TFRepr.
*/

namespace fasst {
	class ERBLETRepr : public Abstract_TFRepr {
	public:
		// ------------ Constructor ------------
		/*!
		The main constructor of the class.
		It initializes the Fourier transform of the analysis window given the sampling frequency,
		the number of time samples of the signal and the number of bin per ERB.
		It initializes the center frequency and the ERB scale.
		It initializes the sythesis window
		\param nOrigSamples number of time samples
		\param wlen the window length _ie._ the length (in audio samples) of one time frame
        \param fs the sampling frequency
		\param binPerERB number of frequency bin per ERB
		*/
		ERBLETRepr(const int nOrigSamples, const int wlen, const int fs, const int binPerERB);

		// ------------ Methods ------------
		/*!
		This method computes the ERBLET direct transform of an audio signal.
		\param x Audio signal
		\return The half ERBLET transform of x (not mirrored) of size (F x 1) x (Mf x I) with
		F: The number of frequency bins
		Mf: The filter size for a given f
		I: The number of channels
		*/
		ArrayMatrixXcd direct(const Audio & x) const ;

        /*!
        This method compute the ERBLET direct transform + framing in each subbands
        \param x Audio signal
        \return the framed transformed signal : (N x F) x (wlenBand(f) x I) with
		N: The number of time frames
		F: The number of frequency bins
		wlenBand(f): wlen for the frequency band f
		I: The number of channels
        */
        ArrayMatrixXcd directFraming(const Audio & x) const override;

		/*!
		This method computes the ERBLET inverse transform of a transformed signal X.
		The output has the following size:
		TODO: explicit the size
		\param X The half transformed signal in the ERB domain : (F x 1) x (Mf x I) with
		F: The number of frequency bins
		Mf: The filter size for a given f
		I: The number of channels
		\return the signal in the time domain : (m_origSamples x I)
		*/
		Eigen::ArrayXXd inverse(const ArrayMatrixXcd & X) const override;

		// ------------ Accessors ------------
		/*!
		This method returns the ERB scale (the bandwith centered at each central frequency).
		It implements the following equation:
		ERB(F) = 24.7 + F/9.625
		\return the erb scale (in Hz)
		*/
		inline VectorXd erb_scale() const { return m_gamma; };

		/*!
		This method returns central frequency of each ERB (in Hz)
		\return Central frequency of each ERB (in Hz)
		*/ 
		inline VectorXd central_freq() const { return m_fc; };

		/*!
		This method returns the filter size at a frequency bin
		*/
		inline int filterSize(int bin) const { return m_M(bin); }

		/*!
		This method returns the analysis filters
		\return The analysis filters
		*/
		inline VectorVectorXd analysisFilter() const { return m_g_an; };

		/*!
		This method returns the synthesis filters
		\return The synthesis filters
		*/
		inline VectorVectorXd synthesisFilter() const { return m_g_syn; };

		/*!
		This method computes the ERBLET direct transform, then frame the transform, then applies the Wiener filter, then computes the inverse ERBLET transform.
		\param x the mixture audio signal
		\param wiener A pointer to an object which implements Wiener_Interface
		\dirname The directory where the filtered sources are saved
		\return the audio signal of each source
		*/
		void Filter(const Audio & x, Wiener_Interface * wiener, const std::string & dirName) override ;

	private:
		//  ----- Members (initialized by the constructor) -----
		int m_fullBins;         // 2*m_bins - 1
		VectorXi m_M;           // Number of time samples in each channel - Size m_bins
		VectorXi m_shift;       // Frequency shifts - Size m_fullBins
        VectorXi m_posit;       // rounded central frequencies of all bins
		VectorVectorXd m_g_an;  // Analysis filter g - Size m_bins
		VectorVectorXd m_g_syn; // Synthesis filter g - Size m_fullBins
		VectorXd m_gamma;       // ERB scale - Size m_bins
		VectorXd m_fc;          // Central frequency of each ERB (in Hz) - Size m_bins

        // ------------ Methods ------------
		VectorXd firwin(int x);
		void nsgabdual_painless();
		VectorXd nsgabframediag(const VectorXi timepos);

		// Cumulative sum (row wise)
		template <typename DerivedIn>
		DerivedIn cumsum(const DerivedIn in) const;

		// Circular shifting (row wise)
		template <typename DerivedIn> 
		DerivedIn circshift(const DerivedIn & in, int k) const;

		// Modulo 
		template <typename DerivedIn>
		DerivedIn modulo(const DerivedIn & in, int k) const;
	};
}
#endif