// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "fasst/Audio.h"
#include "fasst/XMLDoc.h"
#include "fasst/MixCovMatrix.h"
#include "fasst/InputParser.h"
#include <iostream>

#include "fasst/Sources.h"

using namespace std;
using namespace fasst;

int main(int argc, char *argv[]) {

  try {
      // Read command line args
      InputParser input(argc, argv);
      string exeName = input.getArg(0);
      string inputWavFile = input.getArg(1);
      string inputXMLFile = input.getArg(2);
      string outputBinPath = input.getArg(3);

      // Show help if missed input arg or -h option
      if (input.cmdOptionExists("-h") | inputWavFile.empty() | inputXMLFile.empty() | outputBinPath.empty()) {
          cout << "Usage:\t" << exeName;
          cout << " input-wav-file input-xml-file output-bin-path" << endl;
          cout << "\t# input-wav-file : Input audio mixture (.wav)" << endl;
          cout << "\t# input-xml-file : Input sources information file name (.xml)" << endl;
          cout << "\t# output-bin-path : Output path " << endl;
          return 1;
      }
	  // Read audio
	  fasst::Audio x(inputWavFile.c_str());

	  // Read TFR parameters from XML
	  fasst::XMLDoc doc(inputXMLFile.c_str());

	  std::string tfr_type = doc.getTFRType();

	  int wlen = doc.getWlen();
	  int nbin = doc.getNbin();
	  int nbinPerERB = doc.getNbinPerERB();

	  // Get current sources
	  fasst::Sources sources = doc.getSources();

	  // Compute Rx
	  fasst::MixCovMatrix Rx(x, tfr_type, wlen, nbin, nbinPerERB);

	  // Export Rx
	  Rx.write(outputBinPath);

	  return 0;
  }
  catch (exception& e) {
	  cout << e.what();
	  return 1;
  }

}
