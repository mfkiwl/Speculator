// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "fasst/XMLDoc.h"
#include "fasst/Sources.h"
#include "fasst/MixCovMatrix.h"
#include "fasst/NaturalStatistics.h"
#include "fasst/InputParser.h"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace Eigen;
using namespace tinyxml2;
using namespace fasst;

int main(int argc, char *argv[]) {

    try {
        // Read command line args
        InputParser input(argc, argv);
        string exeName = input.getArg(0);
        string inputXMLFile = input.getArg(1);
        string inputRxPath = input.getArg(2);
        string outputXMLFile = input.getArg(3);

        // Show help if missed input arg or -h option
        if (input.cmdOptionExists("-h") | inputXMLFile.empty() | inputRxPath.empty() | outputXMLFile.empty()) {
            cout << "Usage:\t" << exeName;
            cout << " input_XML_File input_bin_file output_XML_file [-NoSimAnn]" << endl;
            cout << "\t# input_XML_File : Input sources information file name (.xml)" << endl;
            cout << "\t# input_bin_path : Path to Rx.bin and Rx_en.bin files " << endl;
            cout << "\t# output_XML_file : Output sources information (updated) file name (.xml)" << endl;
            cout << "\t# -NoSimAnn : Optional param. Set this parameter to disable simulated annealing " << endl;
            return 1;
        }
        bool noSimAnn = input.cmdOptionExists("-NoSimAnn");
        if (noSimAnn) {
            cout << "Disable simulated annealing" << endl;
        }
        else {
            cout << "Enable simulated annealing" << endl;
        }

        // Load sources
        fasst::XMLDoc doc(inputXMLFile.c_str());
        fasst::Sources sources = doc.getSources();

        // Load Rx and Rx_en into hatRx object
        fasst::MixCovMatrix hatRx;
        hatRx.read(inputRxPath, "Rx");
        hatRx.read(inputRxPath, "Rx_en");

        int F = hatRx.bins();
        int N = hatRx.frames();
        int I = hatRx.channels();

        // Check if dimensions are consistent
        if (F != sources.bins()) {
            cout << "Error:\tnumber of bins is not consistent:\n";
            cout << "F = " << F << " in Rx found at " << inputRxPath << '\n';
            cout << "F = " << sources.bins() << " in " << argv[1] << '\n';
            return 1;
        }
        if (N != sources.frames()) {
            cout << "Error:\tnumber of frames is not consistent:\n";
            cout << "N = " << N << " in Rx found at " << inputRxPath << '\n';
            cout << "N = " << sources.frames() << " in " << argv[1] << '\n';
            return 1;
        }
        if (I != sources.channels()) {
            cout << "Error:\tnumber of channels is not consistent:\n";
            cout << "I = " << I << " in Rx found at " << inputRxPath << '\n';
            cout << "I = " << sources.channels() << " in " << argv[1] << '\n';
            return 1;
        }

      // Compute additive noise
	  VectorXd noise_beg = hatRx.getRxEnergy() / 100;
	  VectorXd noise_end = hatRx.getRxEnergy() / 10000;

	  // Set additive noise to the sources 
	  sources.setNoiseBeg(noise_beg);
	  sources.setNoiseEnd(noise_end);

	  // Define number of iterations
	  int iterations = doc.getIterations();

      // Pre EM normalization
      cout << "Pre EM parameter normalization" << endl;
      sources.preEmNormalization(hatRx);

	  // Main loop
	  NaturalStatistics stats(sources, iterations);

	  for (int iter = 0; iter < iterations; iter++) {
		  cout << "      GEM iteration " << iter + 1 << "/" << iterations << '\t';

		  // Conditional expectation of the natural statistics and log-likelihood
		  stats.EStep(sources, hatRx, noSimAnn,iter);

		  double log_like = stats.logLikelihood();
		  if (iter == 0)
			  cout << "Log-likelihood: " << log_like << endl;
		  else {
			  cout << "Log-likelihood: " << log_like << '\t';
			  cout << "Improvement: " << stats.logLikelihoodImprovement() << endl;
		  }

		  // Update A (global)
		  stats.updateMixingParameter();

		  // Update V
		  sources.updateSpectralPower(stats);
	  }

	  // Update each source mixing parameter from the global mixing parameter
	  sources.updateSourceMixingParameter(stats.A_global());

	  // Update the current doc with new sources, then save
	  doc.updateXML(sources);
	  doc.write(outputXMLFile.c_str());
	  return 0;
  }
  catch (exception& e) {
	  cout << e.what();
	  return 1;
  }

}
