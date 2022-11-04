#include "fasst/Audio.h"
#include "fasst/XMLDoc.h"
#include "fasst/TFRepr.h"
#include "fasst/Sources.h"
#include "fasst/InputParser.h"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <chrono>

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
            cout << " input_wav_file input_xml_file input_Rx_file output_wav_dir" << endl;
            cout << "\t# input_wav_file : Input audio mixture (.wav)" << endl;
            cout << "\t# input_xml_file : Input sources information file name (.xml)" << endl;
            cout << "\t# input_Rx_path : Path to Rx.bin and Rx_en.bin" << endl;
            cout << "\t# output_wav_dir : Output directory for sources estimation (.wav) " << endl;
			cout << "\t# [-oTime] : Enable this option to display Wiener Filter execution time in console" << endl;
			cout << "\t# [-oSave output.txt] : Save Wiener Filter execution times in output.txt (this option enable -oTime)" << endl;
			return 1;
		}

		// Test if write out perf. time results
		ofstream outFile;
		bool saveCompTime = input.cmdOptionExists("-oSave");
		bool timeComputation = saveCompTime | input.cmdOptionExists("-oTime");

		string fPerfOutName = "";
		if (saveCompTime) {
			fPerfOutName = input.getCmdOption("-oSave");
			outFile.open(fPerfOutName, std::ofstream::out | std::ofstream::trunc);
			outFile << "Wiener Time Exe (ms)" << endl;
		}

		std::chrono::high_resolution_clock::time_point wienerStart;
		std::chrono::high_resolution_clock::time_point wienerEnd;

		// Read audio
		fasst::Audio x(inputWavFile.c_str());

		// Read wlen and TFR type from XML
		fasst::XMLDoc doc(inputXMLFile.c_str());
		std::string tfr_type = doc.getTFRType();
		int wlen = doc.getWlen();

		// Read output dirname
		string dirname = outputWavDir;
		if (dirname[dirname.length() - 1] != '/') {
			dirname.push_back('/');
		}

		// Load sources
		fasst::Sources sources = doc.getSources();
		int J = sources.size();

        // Load Rx_en and compute noise vector
        fasst::MixCovMatrix hatRx;
        hatRx.read(inputRxPath, "Rx_en");
        VectorXd noise_end = hatRx.getRxEnergy() / 10000;

		// Wiener filter + write audio file
		if (timeComputation) {
			wienerStart = std::chrono::high_resolution_clock::now();
		}
		sources.Filter(x, tfr_type, wlen, noise_end, dirname);
		if (timeComputation) {
			wienerEnd = std::chrono::high_resolution_clock::now();
		}

		if (timeComputation) {
			int wienerDur = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(wienerEnd - wienerStart).count());
			cout << "Wiener Filter time processing : " << wienerDur << " ms\n";
			// Write out
			if (outFile.is_open()) {
				outFile << wienerDur << " \n ";
				outFile.close();
			}
		}

		return 0;
	}
	catch (exception& e) {
		cout << e.what();
		return 1;
	}
}
