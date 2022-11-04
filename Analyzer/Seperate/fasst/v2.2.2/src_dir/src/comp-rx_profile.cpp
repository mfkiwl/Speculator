#include "fasst/Audio.h"
#include "fasst/XMLDoc.h"
#include "fasst/MixCovMatrix.h"
#include "fasst/InputParser.h"
#include <iostream>
#include "fasst/Sources.h"
#include <fstream>
#include <chrono>


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
			cout << " input-wav-file input-xml-file output-bin-file" << endl;
			cout << "\t# input-wav-file : Input audio mixture (.wav)" << endl;
			cout << "\t# input-xml-file : Input sources information file name (.xml)" << endl;
            cout << "\t# output-bin-path : Output path " << endl;
			cout << "\t# [-oTime] : Enable this option to compute Rx execution time in console" << endl;
			cout << "\t# [-oSave output.txt] : Save Rx execution time in output.txt (this option enable -oTime)" << endl;
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
			outFile << "Rx time processing (ms)" << endl;

		}

		std::chrono::high_resolution_clock::time_point rxStart;
		std::chrono::high_resolution_clock::time_point rxEnd;

		// Read audio
		fasst::Audio x(inputWavFile.c_str());

		// Read TFR parameters from XML
		fasst::XMLDoc doc(inputXMLFile.c_str());

		std::string tfr_type = doc.getTFRType();

		int wlen = doc.getWlen();
		int nbin = doc.getNbin();

		// Get current sources
		fasst::Sources sources = doc.getSources();

		// Compute Rx
		if (timeComputation) {
			rxStart = std::chrono::high_resolution_clock::now();
		}
		fasst::MixCovMatrix Rx(x, tfr_type, wlen, nbin);
		if (timeComputation) {
			rxEnd = std::chrono::high_resolution_clock::now();
		}

		if (timeComputation) {
			int rxDur = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(rxEnd - rxStart).count());
			cout <<"Rx time processing : " << rxDur << " ms\n";
			// Write out
			if (outFile.is_open()) {
				outFile << rxDur << " \n ";
				outFile.close();
			}
		}
		// Export Rx
        Rx.write(outputBinPath);

		return 0;
	}
	catch (exception& e) {
		cout << e.what();
		return 1;
	}

}
