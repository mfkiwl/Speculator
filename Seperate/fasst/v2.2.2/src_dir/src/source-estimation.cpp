// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "fasst/Audio.h"
#include "fasst/XMLDoc.h"
#include "fasst/STFTRepr.h"
#include "fasst/Sources.h"
#include "fasst/ERBLETRepr.h"
#include "fasst/ERBRepr.h"
#include "fasst/InputParser.h"
#include "fasst/MixCovMatrix.h"
#include <Eigen/Dense>
#include <iostream>
#include <memory>

using namespace std;
using namespace Eigen;
using namespace fasst;

int main(int argc, char *argv[]) {

  try {
      // Read command line args
      InputParser input(argc, argv);
      string exeName = input.getArg(0);
      string inputWavFile = input.getArg(1);
      string inputXMLFile = input.getArg(2);
      string inputRxPath = input.getArg(3);
      string outputWavDir = input.getArg(4);

      // Show help if missed input arg or -h option
      if (input.cmdOptionExists("-h") | inputWavFile.empty() | inputXMLFile.empty() | inputRxPath.empty() | outputWavDir.empty()) {
          cout << "Usage:\t" << exeName;
          cout << " input_wav_file input_xml_file input_Rx_path output_wav_dir" << endl;
          cout << "\t# input_wav_file : Input audio mixture (.wav)" << endl;
          cout << "\t# input_xml_file : Input sources information file name (.xml)" << endl;
          cout << "\t# input_Rx_path : Path to Rx.bin and Rx_en.bin" << endl;
          cout << "\t# output_wav_dir : Output directory for sources estimation (.wav) " << endl;
          return 1;
      }

	  // Read audio
	  fasst::Audio x(inputWavFile.c_str());

	  // Read wlen and TFR type from XML
	  fasst::XMLDoc doc(inputXMLFile.c_str());
	  std::string tfr_type = doc.getTFRType();
	  int wlen = doc.getWlen();
	  int binPerERB = doc.getNbinPerERB();

	  // Read output dirname
	  string dirname = outputWavDir;
	  if (dirname[dirname.length() - 1] != '/') {
		  dirname.push_back('/');
	  }

	  // Load sources
	  fasst::Sources  sources = fasst::Sources(doc.getSources());
	  //*sources = doc.getSources();
      
      // Load Rx_en and compute noise vector
      fasst::MixCovMatrix hatRx;
      hatRx.read(inputRxPath,"Rx_en");
      
      VectorXd noise_end = hatRx.getRxEnergy() / 10000;

	  // Set sources noise end
	  sources.setNoiseEnd(noise_end);

	  // (optionnal) 
	  sources.spectralPowerSmoothing();

	  // Wiener filter + write audio
	 
	  if (tfr_type == "ERB") {
		  //sources.Filter(x, tfr_type, wlen, noise_end, dirname, nBinPerERB);
		  ERBRepr::FilterERB(x, wlen,&sources, dirname);
	  } 
	  else if (tfr_type == "ERBLET") {
		  std::unique_ptr<Abstract_TFRepr> myRepr(new ERBLETRepr(x.samples(), wlen, x.samplerate(), binPerERB));
		  myRepr->Filter(x, &sources, dirname);
	  }
	  else if (tfr_type == "STFT")
	  {
		  std::unique_ptr<Abstract_TFRepr> myRepr(new STFTRepr(x.samples(), wlen));
		  myRepr->Filter(x, &sources, dirname);
	  }
	  return 0;
  }
  catch (exception& e) {
	  cout << e.what();
	  return 1;
  }
}
