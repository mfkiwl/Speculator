// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "MixCovMatrix.h"
#include "STFTRepr.h"
#include "ERBRepr.h"
#include "ERBLETRepr.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

using namespace std;
using namespace Eigen;

namespace fasst {

    MixCovMatrix::MixCovMatrix(const Audio &x, std::string tfr_type, int wlen, int nbin, int nbinperERB) : m_Rx_name("Rx.bin"), m_Rx_energy_name("Rx_en.bin") {
        if (tfr_type == "STFT") {
            // Compute time-frequency representation
			STFTRepr STFT(x.samples(), wlen);
            ArrayMatrixXcd X = STFT.directFraming(x);
            int F = STFT.bins();
            int N = STFT.frames();

            // Compute covariance matrix
            m_Rx = ArrayMatrixXcd(F, N);
            for (int f = 0; f < F; f++) {
                for (int n = 0; n < N; n++) {
                    m_Rx(f, n) =  X(f, n).adjoint() * X(f, n);
                }
            }
        }
		else if (tfr_type == "ERBLET") {
			ERBLETRepr ERB(x.samples(), wlen, x.samplerate(), nbinperERB);
			ArrayMatrixXcd X = ERB.directFraming(x);

			int N = ERB.frames();
			int F = ERB.bins();

			m_Rx = ArrayMatrixXcd(F, N);

			for (int f = 0; f < F; f++) {
				for (int n = 0; n < N; n++) {
					m_Rx(f, n) = X(n, f).adjoint() * X(n, f);
				}
			}

		}
        else if (tfr_type == "ERB") {
            m_Rx = ERBRepr(x, wlen, nbin).get();
        }
        else {
            stringstream s;
            s << "Wrong TFR type" << tfr_type << ".";
            throw runtime_error(s.str());
        }

        // Compute Rx_energy
        compRxEnergy();
    }
    
    MixCovMatrix::MixCovMatrix(const std::string path) : m_Rx_name("Rx.bin"), m_Rx_energy_name("Rx_en.bin") {
        read(path, "Rx");
        read(path, "Rx_en");
    }
    
    void MixCovMatrix::read(string path,string mat) {

        if (mat == "Rx") {
            // Open fname
            ifstream in((path + m_Rx_name).c_str(), ios_base::binary);
            if (!in.good()) {
                stringstream s;
                s << "Can not open " << (path + m_Rx_name).c_str() << ". ";
                s << "File probably doesn't exist or isn't readable.";
                throw runtime_error(s.str());
            }

            // Read ndim
            int ndim = 0;
            in.read(reinterpret_cast<char *>(&ndim), sizeof(int));

            // Read dim
            vector<int> dim(ndim);
            in.read(reinterpret_cast<char *>(&dim[0]), sizeof(int) * ndim);

            int ndata = 1;
            for (int i = 0; i < ndim; i++) {
                ndata *= dim[i];
            }

            // Read data to a buffer
            vector<float> data(ndata);
            in.read(reinterpret_cast<char *>(&data[0]), sizeof(float) * ndata);
            in.close();

            int I = static_cast<int>(std::sqrt(static_cast<double>(dim[0])));
            int F = dim[1];
            int N = dim[2];

            // Load buffer
            m_Rx = ArrayMatrixXcd(F, N);
            for (int n = 0; n < N; n++) {
                for (int f = 0; f < F; f++) {
                    MatrixXcd Rx_fn(I, I);

                    // Real diagonal elements
                    int ind1 = f * I * I + n * F * I * I;
                    for (int i = 0; i < I; i++) {
                        Rx_fn(i, i) = data[i + ind1];
                    }

                    // Complex elements
                    int sum = 0;
                    for (int i1 = 0; i1 < I - 1; i1++) {
                        for (int i2 = i1 + 1; i2 < I; i2++) {
                            int ind2 = ind1 + (i2 - i1 + sum) * 2 - 2 + I;
                            Rx_fn(i1, i2) = complex<double>(data[ind2], data[ind2 + 1]);
                            Rx_fn(i2, i1) = conj(Rx_fn(i1, i2));
                        }
                        sum += I - 1 - i1;
                    }
                    m_Rx(f, n) = Rx_fn;
                }
            }
        }
        else if (mat == "Rx_en") {
            // Open fname
            ifstream in((path + m_Rx_energy_name).c_str(), ios_base::binary);
            if (!in.good()) {
                stringstream s;
                s << "Can not open " << (path + m_Rx_name).c_str() << ". ";
                s << "File probably doesn't exist or isn't readable.";
                throw runtime_error(s.str());
            }

            // Read ndim
            int ndim = 0;
            in.read(reinterpret_cast<char *>(&ndim), sizeof(int));

            // Read dim
            vector<int> dim(ndim);
            in.read(reinterpret_cast<char *>(&dim[0]), sizeof(int) * ndim);

            int ndata = 1;
            for (int i = 0; i < ndim; i++) {
                ndata *= dim[i];
            }

            // Read data to a buffer
            vector<double> data(ndata);
            in.read(reinterpret_cast<char *>(&data[0]), sizeof(double) * ndata);
            in.close();
            
            // Load buffer
            int F = dim[0];
            m_Rx_energy = VectorXd(F);
            m_Rx_energy = Eigen::Map<Eigen::VectorXd, Eigen::Aligned>(data.data(), data.size());
        }
        else {
            stringstream s;
            s << "Mat to load unkown. Possible choices : Rx or Rx_en.";
            throw runtime_error(s.str());
        }
    }

    void MixCovMatrix::compRxEnergy()  {
        int F = bins();
        int N = frames();
        int I = channels();

        m_Rx_energy = VectorXd::Zero(F);
        for (int f = 0; f < F; f++) {
            for (int n = 0; n < N; n++) {
                m_Rx_energy(f) += (m_Rx(f, n).real().trace() / I);
            }
            m_Rx_energy(f) /= N;
        }
    }

    void MixCovMatrix::write(string path) {
        int F = bins();
        int N = frames();
        int I = channels();

        // ---- Write out Rx ----

        // Open
        ofstream outRx((path + m_Rx_name).c_str(), ios_base::binary);
        if (!outRx.good()) {
            stringstream s;
            s << "Can not open " << (path + m_Rx_name).c_str() << ". ";
            s << "You probably don't have write access to this location.";
            throw runtime_error(s.str());
        }

        // Write ndim
        int ndim = 3;
        outRx.write(reinterpret_cast<char *>(&ndim), sizeof(int));

        // Write dim
        vector<int> dim(ndim);
        dim[0] = I * I;
        dim[1] = F;
        dim[2] = N;
        outRx.write(reinterpret_cast<char *>(&dim[0]), sizeof(int) * ndim);

        // Load to buffer
        int ndata = I * I * F * N;
        vector<float> data(ndata);

        for (int n = 0; n < N; n++) {
            for (int f = 0; f < F; f++) {
                int ind1 = (f * I * I) + (n * F * I * I);
                // Real diagonal elements
                for (int i = 0; i < I; i++) {
                    data[i + ind1] = static_cast<float>(m_Rx(f, n)(i, i).real());
                }
                // Complex elements
                int sum = 0;
                for (int i1 = 0; i1 < I - 1; i1++) {
                    for (int i2 = i1 + 1; i2 < I; i2++) {
                        int ind2 = ind1 + (i2 - i1 + sum) * 2 - 2 + I;
                        data[ind2] = static_cast<float>(m_Rx(f, n)(i1, i2).real());
                        data[ind2 + 1] = static_cast<float>(m_Rx(f, n)(i1, i2).imag());
                    }
                    sum += I - 1 - i1;
                }
            }
        }

        // Write buffer to file
        outRx.write(reinterpret_cast<char *>(&data[0]), sizeof(float) * ndata);
        outRx.close();

        // ---- Write out Rx_energy ----
        // Open
        ofstream outRxEn((path + m_Rx_energy_name).c_str(), ios_base::binary);
        if (!outRxEn.good()) {
            stringstream s;
            s << "Can not open " << (path + m_Rx_name).c_str() << ". ";
            s << "You probably don't have write access to this location.";
            throw runtime_error(s.str());
        }

        // Write ndim
        ndim = 1;
        outRxEn.write(reinterpret_cast<char *>(&ndim), sizeof(int));

        // Write dim
        dim.clear();
        dim.resize(ndim);
        dim[0] = F;
        outRxEn.write(reinterpret_cast<char *>(&dim[0]), sizeof(int) * ndim);

        // Load to buffer
        ndata = F;
        vector<double> data2(ndata);
        VectorXd::Map(&data2[0], m_Rx_energy.size()) = m_Rx_energy;
        outRxEn.write(reinterpret_cast<char *>(&data2[0]), sizeof(double) * ndata);
        outRxEn.close();
    }

    bool MixCovMatrix::equal_Rx(const MixCovMatrix & m, double margin) const {
        cout << "Compare Rx " << endl;
        bool res;
        compare((*this).getRx(), m.getRx(), margin) == 0 ? res=true : res=false;
        return res;
    }

    bool MixCovMatrix::equal_Rx_en(const MixCovMatrix & m, double margin) const {
        cout << "Compare Rx energy " << endl;
        bool res;
        compare((*this).getRxEnergy(), m.getRxEnergy(), margin) == 0 ? res = true : res = false;
        return res;
        
    }

	bool  MixCovMatrix::equal(const MixCovMatrix & m, double margin) const {
		return (*this).equal_Rx(m, margin) && (*this).equal_Rx_en(m, margin);
	}
}
