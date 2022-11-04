// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

/*
In order to execute this code in debug to compare c++ and Matlab output, please add the following properties to the project:

Property->Debug->Environment: 
PATH=$(VCRedistPaths)%PATH%$(LocalDebuggerEnvironment);C:\Program Files\MATLAB\R2017a\bin\win64

Property->C/C++->General->include: 
C:\Program Files\MATLAB\R2017a\extern\include

Property->Linker->Entry:
C:\Program Files\MATLAB\R2017a\extern\lib\win64\microsoft\libmx.lib
C:\Program Files\MATLAB\R2017a\extern\lib\win64\microsoft\libmex.lib
C:\Program Files\MATLAB\R2017a\extern\lib\win64\microsoft\libmat.lib

Then, set the MATLAB_LIBRARY_LINKING preprocessing value to 1
*/

#define MATLAB_LIBRARY_LINKING 0

#include "../fasst/ERBLETRepr.h"
#include "../fasst/Audio.h"
#include "../fasst/typedefs.h"
#include "gtest/gtest.h"
#include <Eigen/Core>

#if MATLAB_LIBRARY_LINKING
#include "mat.h"
#include "matrix.h"
#endif

using namespace std;
using namespace Eigen;

string g_inputDataDir(INPUT_DATA_DIR_ERBLET);

TEST(ERBLETRepr, dev) {

	string fname(g_inputDataDir + "/RWC_pop_16000Hz_stereo_10sec.wav");
	double SNR_ref = 295; // 300 dB
	int nKeepSamples = 16000;

	fasst::Audio x(fname.c_str());

	int fs = x.samplerate();
	int nSamples = x.samples();

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

	ASSERT_GT(SNR, SNR_ref);

}

#if MATLAB_LIBRARY_LINKING
/*
This function read a configuration contained in a .mat file
*/
void read_config(const string gPath, int & fs, int & wlen, int & nSamplesOrig, int & binsPerERB);

/*
This function read a .mat file containing the varName cell array of double
*/
ArrayMatrixXd read_double_cellArray(const string gPath, const string varName);

/*
This function read a .mat file containing the varName cell array of complex number
*/
ArrayMatrixXcd read_complex_cellArray(const string gPath, const string varName);

/*
This function compare the analysis or synthesis filters between the C++ and the matlab version
with a given precision
*/
template<typename T1, typename T2>
void compare(T1 mat_ref, T2 c_ref, const double precision);

/*!
This test compare the Matlab and c++ generated analysis and synthesis filters
*/

TEST(ERBLETRepr,compare_filter) {
  double precision = std::pow(10, -16);
  string gPath = "C:\\FASST_GIT\\dev\\build_dir\\";
  vector<string> fname{"16000_1024_32589_1.mat","16000_1024_32589_6.mat","16000_1024_32589_12.mat",
	  "16000_1024_77889_1.mat","16000_1024_77889_6.mat","16000_1024_77889_12.mat"};

  vector<string> varName{"g_an","g_syn"};

  for(int f = 0; f < fname.size(); f++) {
	  cout << "---- File : " + fname[f] << "----" << endl;
	  int fs, nSamplesOrig, wlen, binsPerERB;
	  string file = gPath + fname[f];

	  // read config
	  read_config(file, fs, wlen, nSamplesOrig, binsPerERB);

	  // Create the ERB
	  fasst::ERBLETRepr ERB(nSamplesOrig, wlen, fs, binsPerERB);

	  // Get the c++ and matlab analysis filters
	  {
		  VectorVectorXd an_c = ERB.analysisFilter();
		  ArrayMatrixXd an_mat =  read_double_cellArray(file, string("g_an"));
		  cout << "Compare analysis filter - precision " << precision << endl;
		  ArrayMatrixXd an_cc(an_c.size(),1);
		  for (int i = 0; i < an_cc.size(); i++) {
			  an_cc(i,0) = MatrixXd(an_c[i].size(), 1);
			  an_cc(i,0).col(0) = an_c[i];
		  }

		  compare(an_mat, an_cc, precision);
	  }
	  
	  {
		  VectorVectorXd syn_c = ERB.synthesisFilter();
		  ArrayMatrixXd syn_mat = read_double_cellArray(file, string("g_syn"));
		  cout << "Compare synthesis filter - precision " << precision << endl;
		  
		  ArrayMatrixXd syn_cc(syn_c.size(),1);
		  for (int i = 0; i < syn_c.size(); i++) {
			  syn_cc(i, 0) = MatrixXd(syn_c[i].size(), 1);
			  syn_cc(i, 0).col(0) = syn_c[i];
		  }
		  compare(syn_mat, syn_cc, precision);
	  }

  }

}

/*!
This test compare the the c++ direct transform output with the matlab one
*/

TEST(ERBLETRepr, direct_transform) {
	double precision = std::pow(10, -12);
	string gPath = "C:\\FASST_GIT\\dev\\build_dir\\";

	string wavFile = "RWC_pop_16000Hz_stereo_10sec.wav";
	vector<string> fname{ "16000_1024_32589_1.mat","16000_1024_32589_6.mat","16000_1024_32589_12.mat",
		"16000_1024_77889_1.mat","16000_1024_77889_6.mat","16000_1024_77889_12.mat" };


	for (int f = 0; f < fname.size(); f++) {
		cout << "---- File : " + fname[f] << "----" << endl;
		int fs, nSamplesOrig, wlen, binsPerERB;
		string file = gPath + fname[f];

		// read config
		read_config(file, fs, wlen, nSamplesOrig, binsPerERB);

		// Create the ERB
		fasst::ERBLETRepr ERB(nSamplesOrig, wlen, fs, binsPerERB);

		// read the Matlab direct transform
		Eigen::Matrix<Eigen::Matrix<Eigen::dcomplex, -1, -1>, -1, -1> X_mat = read_complex_cellArray(file, string("X"));

		// Get the c++ direct transform
		fasst::Audio x = fasst::Audio((gPath + wavFile).c_str());

		// Keep nSampleOrig
		ArrayXXd xKept = x.topRows(nSamplesOrig);
		x = fasst::Audio(xKept, fs);

		// Apply direct transform
		ArrayMatrixXcd X_c = ERB.direct(x);
		
		// Compare the direct transformed signal between c and matlab
		compare(X_mat, X_c, precision);
	}

}


/*!
This test apply the direct transform and the framing and compare with the Matlab ref
*/

TEST(ERBLETRepr, direct_framed_transform) {
	double precision = std::pow(10, -12);
	string gPath = "C:\\FASST_GIT\\dev\\build_dir\\";

	string wavFile = "RWC_pop_16000Hz_stereo_10sec.wav";
	vector<string> fname{ "16000_1024_32589_1.mat","16000_1024_32589_6.mat","16000_1024_32589_12.mat",
		"16000_1024_77889_1.mat","16000_1024_77889_6.mat","16000_1024_77889_12.mat" };


	for (int f = 0; f < fname.size(); f++) {
		cout << "---- File : " + fname[f] << "----" << endl;
		int fs, nSamplesOrig, wlen, binsPerERB;
		string file = gPath + fname[f];

		// read config
		read_config(file, fs, wlen, nSamplesOrig, binsPerERB);

		// Create the ERB
		fasst::ERBLETRepr ERB(nSamplesOrig, wlen, fs, binsPerERB);

		// read the Matlab direct transform
		ArrayMatrixXcd Xfram_mat = read_complex_cellArray(file, string("Xfram"));

		// Get the c++ direct transform
		fasst::Audio x = fasst::Audio((gPath + wavFile).c_str());

		// Keep nSampleOrig
		ArrayXXd xKept = x.topRows(nSamplesOrig);
		x = fasst::Audio(xKept, fs);

		ArrayMatrixXcd Xfram_c = ERB.directFraming(x);

		// Compare the direct transformed signal between c and matlab
		compare(Xfram_mat, Xfram_c, precision);
	}

}

/*!
This test apply the direct transform then the inverse transform and compute the
reconstruction error as a SNR.
*/
TEST(ERBLETRepr, direct_and_inverse_transform) {

	double SNR_ref = 295; // 300 dB

	string gPath = "C:\\FASST_GIT\\dev\\build_dir\\";
	string wavFile = "RWC_pop_16000Hz_stereo_10sec.wav";
	vector<string> fname{ "16000_1024_32589_1.mat","16000_1024_32589_6.mat","16000_1024_32589_12.mat",
		"16000_1024_77889_1.mat","16000_1024_77889_6.mat","16000_1024_77889_12.mat" };

	for (int f = 0; f < fname.size(); f++) {
		
		cout << "---- File : " + fname[f] << "----" << endl;
		int fs, nSamplesOrig, wlen, binsPerERB;
		string file = gPath + fname[f];

		// read config
		read_config(file, fs, wlen, nSamplesOrig, binsPerERB);

		// Create the ERB
		fasst::ERBLETRepr ERB(nSamplesOrig, wlen, fs, binsPerERB);

		// Get the c++ direct transform
		fasst::Audio x = fasst::Audio((gPath + wavFile).c_str());
		
		// Keep nSampleOrig
		ArrayXXd xKept = x.topRows(nSamplesOrig);
		x = fasst::Audio(xKept,fs);

		// Compute the direct transform
		ArrayMatrixXcd X_c = ERB.direct(x);
		
		// Compute the inverse transform
		ArrayXXd xPrim = ERB.inverse(X_c);

		// Compute the SNR
		double errEn = sqrt((xPrim - x).cwiseAbs2().sum());
		double sigEn = sqrt(x.cwiseAbs2().sum());
		double SNR = 20.0 * log10(sigEn / errEn);

		ASSERT_GT(SNR, SNR_ref);

	}

}


void read_config(const string gPath, int & fs, int & wlen, int & nSamplesOrig, int & binsPerERB)
{
	MATFile *pmat;
	pmat = matOpen(gPath.c_str(), "r");
	if (pmat != NULL) {
		// get fs
		mxArray *arr;
		double   *dataPtr;

		arr = matGetVariable(pmat, "fs");
		dataPtr = (double *)mxGetPr(arr);
		fs = static_cast<int>(*(dataPtr + 0));

		// get wlen
		arr = matGetVariable(pmat, "wlen");
		dataPtr = (double *)mxGetPr(arr);
		wlen = static_cast<int>(*(dataPtr + 0));

		// get nSamplesOrig
		arr = matGetVariable(pmat, "nSamplesOrig");
		dataPtr = (double *)mxGetPr(arr);
		nSamplesOrig = static_cast<int>(*(dataPtr + 0));

		// get binsPerERB
		arr = matGetVariable(pmat, "binsPerERB");
		dataPtr = (double *)mxGetPr(arr);
		binsPerERB = static_cast<int>(*(dataPtr + 0));
		matClose(pmat);
	}

}

ArrayMatrixXcd read_complex_cellArray(const string gPath, const string varName) {
	MATFile *pmat;
	ArrayMatrixXcd out;
	pmat = matOpen(gPath.c_str(), "r");
	if (pmat == NULL) return out;

	// extract the specified variable
	mxArray *arr = matGetVariable(pmat, varName.c_str());

	// Get the number of cells
	//mwSize numCell = mxGetNumberOfElements(arr);
	size_t nDimsCell = mxGetNumberOfDimensions(arr);
	const size_t *cellDims = mxGetDimensions(arr);

	// Init the output
	out = ArrayMatrixXcd(cellDims[0], cellDims[1]);

	for (int cj = 0; cj < cellDims[1]; cj++)
	{
		for (int ci = 0; ci < cellDims[0]; ci++)
		{
			// Get the current cell ptr
			mxArray *cellPtr = mxGetCell(arr, ci + cj * cellDims[0]);

			MatrixXcd mMat;
			if (cellPtr != NULL && mxIsComplex(cellPtr) && !mxIsEmpty(cellPtr)) {
				size_t nDims = mxGetNumberOfDimensions(cellPtr);
				const size_t *dims = mxGetDimensions(cellPtr);
				double   *dataPtr, *dataPti;
				dataPtr = (double *)mxGetPr(cellPtr); // real part
				dataPti = (double *)mxGetPi(cellPtr); // imaj part

													  // Init vMat, the eigen formated vector of matlab values
				mMat = MatrixXcd(dims[0], dims[1]);

				for (int j = 0; j < dims[1]; j++) { // read column per column
					for (int i = 0; i < dims[0]; i++) { // for each row i of column j
						mMat(i, j) = Eigen::dcomplex(*(dataPtr + i + j * dims[0]), *(dataPti + i + j * dims[0]));
					}
				}
			}
			else {
				mMat = MatrixXcd(0, 0);
			}

			// stack current vMat
			out(ci, cj) = mMat;

		}
	}
	matClose(pmat);
	return out;
}

ArrayMatrixXd read_double_cellArray(const string gPath, const string varName) {
	MATFile *pmat;
	ArrayMatrixXd out;
	pmat = matOpen(gPath.c_str(), "r");
	if (pmat == NULL) return out;

	// extract the specified variable
	mxArray *arr = matGetVariable(pmat, varName.c_str());

	// Get the number of cells
	//mwSize numCell = mxGetNumberOfElements(arr);
	size_t nDimsCell = mxGetNumberOfDimensions(arr);
	const size_t *cellDims = mxGetDimensions(arr);

	// Init the output
	out = ArrayMatrixXd(cellDims[0], cellDims[1]);

	for (int cj = 0; cj < cellDims[1]; cj++)
	{
		for (int ci = 0; ci < cellDims[0]; ci++)
		{
			// For cell at (ci,cj)

			// Get the current cell ptr
			mxArray *cellPtr = mxGetCell(arr, ci + cj * cellDims[0]);
			MatrixXd mMat;
			if (cellPtr != NULL && mxIsDouble(cellPtr) && !mxIsEmpty(cellPtr)) {
				// Init vMat, the eigen formated vector of matlab values
				size_t nDims = mxGetNumberOfDimensions(cellPtr);
				const size_t *dims = mxGetDimensions(cellPtr);
				double   *dataPtr;
				dataPtr = (double *)mxGetPr(cellPtr); // real part

													  // Init vMat, the eigen formated vector of matlab values
				mMat = Eigen::Matrix<double, -1, -1>(dims[0], dims[1]);

				for (int j = 0; j < dims[1]; j++) { // read column per column
					for (int i = 0; i < dims[0]; i++) { // for each row i of column j
						mMat(i, j) = *(dataPtr + i + j * dims[0]);
					}
				}
			}
			else {
				mMat = MatrixXd(0, 0);
			}

			// stack current vMat
			out(ci, cj) = mMat;

		}
	}
	matClose(pmat);

	return out;
}


template<typename T1, typename T2>
void compare(T1 mat_ref, T2 c_ref, const double precision) {
	// Check the number of cell rows
	ASSERT_EQ(mat_ref.rows(), c_ref.rows());

	// Check the number of cell cols
	ASSERT_EQ(mat_ref.cols(), c_ref.cols());

	// Check cell content
	int nRows = static_cast<int> (mat_ref.rows());
	int nCols = static_cast<int> (mat_ref.cols());

	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			// Check the number of rows
			ASSERT_EQ(mat_ref(i, j).rows(), c_ref(i, j).rows());

			// Check the number of cols
			ASSERT_EQ(mat_ref(i, j).cols(), c_ref(i, j).cols());

			// Compute the error between c++ and matlab
			ArrayXXd err = (mat_ref(i, j).array() - c_ref(i, j).array()).abs();

			// flag
			ArrayXXi errFlag = (err.array() > precision).cast<int>();

			// Check the number of errors according to the specified precision
			int nErr = (errFlag).sum();

			if (nErr != 0) {
				cout << "Cell (" << i << "," << j << ")" << endl;
				ASSERT_EQ(nErr, 0);
			}

			//cout << "Max error : " << err.maxCoeff() << endl;
		}
	}
}
#endif


