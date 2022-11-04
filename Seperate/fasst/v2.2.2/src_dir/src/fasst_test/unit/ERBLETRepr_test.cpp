// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "../fasst/ERBLETRepr.h"
#include "../fasst/Audio.h"
#include "../fasst/typedefs.h"
#include "../fasst/Wiener_Interface.h"
#include "gtest/gtest.h"
#include <Eigen/Core>


using namespace std;
using namespace Eigen;

string g_inputDataDir(INPUT_DATA_DIR_ERBLET);

TEST(ERBLETRepr, dev) {

	string fname(g_inputDataDir + "/RWC_pop_16000Hz_stereo_10sec.wav");
	double SNR_ref = 300.99; // 300 dB
	int nKeepSamples = 16000;

	fasst::Audio x(fname.c_str());

	int fs = x.samplerate();

	// Keep nSampleOrig
	ArrayXXd xKept = x.topRows(nKeepSamples);
	x = fasst::Audio(xKept, fs);

	int wlen = 1024;
	int ERBperBin = 1;

	// ERBLET class creation.
	fasst::ERBLETRepr ERBLET_Obj(nKeepSamples, wlen, fs, ERBperBin);

	// Direct Transform
	ArrayMatrixXcd X;
	X = ERBLET_Obj.direct(x);

	ArrayXXd xPrim;
	xPrim = ERBLET_Obj.inverse(X);

	ASSERT_EQ(2, x.channels());

	// Compute the SNR
	double errEn = sqrt((xPrim - x).cwiseAbs2().sum());
	double sigEn = sqrt(x.cwiseAbs2().sum());
	double SNR = 20.0 * log10(sigEn / errEn);

	// The abs difference must be lesser than 1
	cout << "SNR: " << SNR << " | SNR ref: " << SNR_ref << endl;
	ASSERT_LT(abs(SNR- SNR_ref), 1.);

}

TEST(ERBLETRepr, filter) {
	int nSamples = 10000;
	int fs = 8000;
	int wlen = 1024;
	int ERBperBin = 1;
	int I = 1;
	int nSrc = 1;
	string saveDir = "./";

	// Create the ERB object
	fasst::ERBLETRepr ERBLET(nSamples, wlen, fs, ERBperBin);

	// Generate an audio of random data
	fasst::Audio x(ArrayXXd::Random(nSamples, I), fs);

	// Declare a wiener concrete class which returns a Wiener gain equal
	// to the identity matrix 
	class Wiener_Concrete : public fasst::Wiener_Interface {
	public:
		Wiener_Concrete(int I,int frames, int bins, int src)
			: m_I(I), m_frames(frames), m_bins(bins), m_src(src)
		{}

		// The only usefull function which return the identity matrix
		Eigen::MatrixXcd computeW(int n, int f, int j) const {
			if (n >= 0 && f >= 0 && j >= 0) // Just to use the input params
				return MatrixXcd::Identity(m_I, m_I);
			else
				return MatrixXcd::Identity(m_I, m_I);
		}

		virtual inline int frames() const { return m_frames; }
		virtual inline int bins() const { return m_bins; }
		virtual inline int sources() const { return m_src; }
		virtual inline int channels() const { return m_I; }
		virtual inline std::string name(int j) const { return ("filtered_" + j); }
	private:
		int m_I,m_frames,m_bins,m_src;

	};

	// Declare a new instance
	Wiener_Concrete W = Wiener_Concrete(I, ERBLET.frames(), ERBLET.bins(), nSrc);

	// Apply Filter
	ERBLET.Filter(x, &W, saveDir);

	// Read the filtered signal 
	string fname_filtered = saveDir + W.name(0) + ".wav";

	// Read the filtered signal
	fasst::Audio x_filt(fname_filtered.c_str());

	//Compare
	ASSERT_LT((x - x_filt).abs().maxCoeff(), 1. / 16369);
		
}
