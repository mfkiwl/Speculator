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
		string inputSrcFile = input.getArg(1);
        string inputRxPath = input.getArg(2);
		string outputSrcFile = input.getArg(3);

		// Show help if missed input arg or -h option
		if (input.cmdOptionExists("-h") | inputSrcFile.empty() | inputRxPath.empty() | outputSrcFile.empty()) {
			cout << "Usage:\t" << exeName;
			cout << " input_XML_File input_bin_file output_XML_file [-Eoptim] [-oTime output.txt]" << endl;
            cout << "\t# input_XML_File : Input sources information file name (.xml)" << endl;
            cout << "\t# input_bin_path : Path to Rx.bin and Rx_en.bin files " << endl;
            cout << "\t# output_XML_file : Output sources information (updated) file name (.xml)" << endl;
			cout << "\t# -NoSimAnn : Optional param. Set this parameter to disable simulated annealing " << endl;
			cout << "\t# [-oTime] : Enable this option to display E and M execution time in console" << endl;
			cout << "\t# [-oSave output.txt] : Save E and M execution times in output.txt (this option enable -oTime)" << endl;

			return 1;
		}
		bool noSimAnn = input.cmdOptionExists("-NoSimAnn");
		if (noSimAnn) {
			cout << "Disable simulated annealing" << endl;
		}
		else {
			cout << "Enable simulated annealing" << endl;
		}

		// Test if write out perf. time results
		ofstream outFile;
		bool saveCompTime = input.cmdOptionExists("-oSave");
		bool timeComputation = saveCompTime | input.cmdOptionExists("-oTime");

		string fPerfOutName = "";
		if (saveCompTime) {
			fPerfOutName = input.getCmdOption("-oSave");
		}

        std::chrono::high_resolution_clock::time_point preEMNormStart;
        std::chrono::high_resolution_clock::time_point preEMNormEnd;
		std::chrono::high_resolution_clock::time_point statsStart;
		std::chrono::high_resolution_clock::time_point statsEnd;
		std::chrono::high_resolution_clock::time_point upMPStart;
		std::chrono::high_resolution_clock::time_point	upMPEnd;
		std::chrono::high_resolution_clock::time_point	upSPStart;
		std::chrono::high_resolution_clock::time_point	upSPEnd;
		unsigned int meanEDur = 0;
		unsigned int meanMSpatDur = 0;
		unsigned int meanMSpecDur = 0;

		// Load sources
		XMLDoc doc(inputSrcFile.c_str());
		Sources sources = doc.getSources();

		// Load hatRx
		MixCovMatrix hatRx;
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

		// Define number of iterations
		int iterations = doc.getIterations();

        // Pre EM normalization
        cout << "Pre EM parameter normalization" << endl;
        if (timeComputation) {
            preEMNormStart = std::chrono::high_resolution_clock::now();
        }
        sources.preEmNormalization(hatRx);
        if (timeComputation) {
            preEMNormEnd = std::chrono::high_resolution_clock::now();
        }

		// Main loop
		NaturalStatistics stats(sources);

		// Test if write out perf. results
		if (saveCompTime) {
			outFile.open(fPerfOutName, std::ofstream::out | std::ofstream::trunc);
			outFile << "#iter \t E (ms) \t M spat (ms) \t M spec (ms)" << endl;
		}


		for (int iter = 0; iter < iterations; iter++) {
			cout << "GEM iteration " << iter + 1 << " of " << iterations << '\t';

			// Compute Sigma_b
			VectorMatrixXcd Sigma_b(F);
            double sigma_f;
			for (int f = 0; f < F; f++) {
                if (noSimAnn) {
                    sigma_f = sqrt(noise_end(f));
                }
                else {
                    sigma_f = (sqrt(noise_beg(f)) * (iterations - iter - 1) +
                        sqrt(noise_end(f)) * (iter + 1)) / iterations;
                }
                Sigma_b(f) = MatrixXcd::Identity(I, I) * sigma_f * sigma_f;
			}

			// Conditional expectation of the natural statistics and log-likelihood
			if (timeComputation) {
				statsStart = std::chrono::high_resolution_clock::now();
			}

			stats.EStep(sources, hatRx, Sigma_b);

			if (timeComputation) {
				statsEnd = std::chrono::high_resolution_clock::now();
			}

			double log_like = stats.logLikelihood();
			if (iter == 0) {
				cout << "Log-likelihood: " << log_like << endl;

			}
			else {
				cout << "Log-likelihood: " << log_like << '\t';
				cout << "Improvement: " << stats.logLikelihoodImprovement() << endl;
			}

			// Update A
			if (timeComputation) {
				upMPStart = std::chrono::high_resolution_clock::now();
			}

			sources.updateMixingParameter(stats);

			if (timeComputation) {
				upMPEnd = std::chrono::high_resolution_clock::now();
			}

			// Update V
			if (timeComputation) {
				upSPStart = std::chrono::high_resolution_clock::now();
			}

			sources.updateSpectralPower(stats);

			if (timeComputation) {
				upSPEnd = std::chrono::high_resolution_clock::now();
			}

			// Compute durations
			if (timeComputation) {
				unsigned int Edur = static_cast<unsigned int>(std::chrono::duration_cast<std::chrono::milliseconds>(statsEnd - statsStart).count());
				unsigned int MSpatDur = static_cast<unsigned int>(std::chrono::duration_cast<std::chrono::milliseconds>(upMPEnd - upMPStart).count());
				unsigned int MSpecDur = static_cast<unsigned int>(std::chrono::duration_cast<std::chrono::milliseconds>(upSPEnd - upSPStart).count());
				// Print out
				cout << "E: " << Edur << " ms\n";
				cout << "M spat: " << MSpatDur << " ms\n";
				cout << "M spec: " << MSpecDur << " ms\n";
				// Write out
				if (outFile.is_open()) {
					outFile << iter << " \t ";
					outFile << Edur << " \t ";
					outFile << MSpatDur << " \t ";
					outFile << MSpecDur << " \t ";
					outFile << endl;
				}
				// Accumulate
				meanEDur += Edur;
				meanMSpatDur += MSpatDur;
				meanMSpecDur += MSpecDur;

			}

		}
		//Average time durations
		if (timeComputation) {
			// Compute and print out average computation times
            unsigned int preEMNormDur = static_cast<unsigned int>(std::chrono::duration_cast<std::chrono::milliseconds>(preEMNormEnd - preEMNormStart).count());
			cout << endl << "E (avg.): " << static_cast<double> (meanEDur / iterations) << " ms\n";
			cout << "M spat (avg.): " << static_cast<double> (meanMSpatDur / iterations) << " ms\n";
			cout << "M spec (avg.): " << static_cast<double> (meanMSpecDur / iterations) << " ms\n";
            cout << endl << "Pre-EM normalization: " << preEMNormDur << "ms \n";
			if (outFile.is_open()) {
				// Compute and write out average computation times
				outFile << endl;
				outFile << "Avg. \t ";
				outFile << static_cast<double> (meanEDur / iterations) << "\t" << static_cast<double> (meanMSpatDur / iterations) << "\t" << static_cast<double> (meanMSpecDur / iterations) << endl;
                outFile << "Pre-EM normalization: " << preEMNormDur << "ms \n";
				outFile.close();
			}

		}

		// Update each source mixing parameter from the global mixing parameter
		sources.updateSourceMixingParameter();

		// Update the current doc with new sources, then save
		doc.updateXML(sources);
		doc.write(outputSrcFile.c_str());
		return 0;
	}
	catch (exception& e) {
		cout << e.what();
		return 1;
	}

}
