// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0
#include "Sources.h"
#include "STFTRepr.h"
#include "ERBRepr.h"
#include "ERBLETRepr.h"
#include "Audio.h"
#include "NaturalStatistics.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

namespace fasst {
Sources::Sources(const XMLElement* xmlNode) {
    // Load sources
    int R = 0;
    // Loop over sources
	int k = 0; // source index
    for (const XMLElement* xmlEl = xmlNode->FirstChildElement("source"); xmlEl != NULL; xmlEl = xmlEl->NextSiblingElement("source"))	{
        Source source(xmlEl,k);
        m_sources.push_back(source);
        R += source.rank();
		k++;
    }
    m_bins = m_sources[0].bins();
    m_frames = m_sources[0].frames();
    m_channels = static_cast<int>(m_sources[0].A(0).rows());
	m_noise_beging = VectorXd::Zero(m_bins);
	m_noise_end = VectorXd::Zero(m_bins);

    // Check if dimensions are consistent for each source
    for (int j = 1; j < sources(); j++) {
        if (m_channels != m_sources[j].A(0).rows()) {
            throw runtime_error(
                "Check your source parameters: number of channels should be the same "
                "for each source mixing parameter");
        }
        if (m_bins != m_sources[j].bins()) {
            throw runtime_error(
                "Check your source parameters: number of rows in Wex and Wft "
                "parameter should be the same for each source");
        }
        if (m_frames != m_sources[j].frames()) {
            throw runtime_error("Check your source parameters: number of columns in "
            "Hex and Hft should be the same for each source");
        }
    }  

}

void Sources::preEmNormalization(const MixCovMatrix &hatRx) {
    int J = this->sources();
    int F = m_bins;
    int N = m_frames;
    int I = m_channels;
    double dataPower = 0;
    double dataPowerChan = 0;

    // Compute dataPower
    for (int f = 0; f < F; f++) {
        dataPowerChan = 0.0;
        for (int n = 0; n < N; n++) {
            for (int i = 0; i < I; i++) {
                dataPowerChan += hatRx(f, n, i, i).real();
            }
        }
        dataPowerChan /= static_cast<double>(I*N*F);
        dataPower += dataPowerChan;
    }
    dataPower = (hatRx.getRxEnergy().sum()) / ( static_cast<double> (F));

    printf("\tData Power : %.10f \n", dataPower);

    // Scale spatial parameter
    double aPower = 0.;
    double vPower = 0.;
    double modelPower = 0.;

    for (int j = 0; j < J; j++) {
		aPower = 0.;
        vPower = 0.;
        int A_rank = m_sources[j].A().rank(); // current source rank

		if (m_sources[j].isInst()) {
			aPower = static_cast<double>(F)*m_sources[j].A(0).array().cwiseAbs2().sum();
		}
		else{
			for (int f = 0; f < F; f++) {
				aPower += m_sources[j].A(f).array().cwiseAbs2().sum();
			}
		}

		aPower /= static_cast<double> (F*I*A_rank);
        //vPower = (m_sources[j].getSPEx().getW() * m_sources[j].getSPEx().getH()).mean(); // for Wex/Hex reprensation only
		vPower = m_sources[j].V().mean(); // cf eq [13]
        modelPower = aPower * vPower;

		if (m_sources[j].isInst()) {
			m_sources[j].A(0) *= sqrt(dataPower / modelPower);
		}
		else {
			for (int f = 0; f < F; f++) {
				m_sources[j].A(f) *= sqrt(dataPower / modelPower);
			}
		}

        printf("\tSource %d -  Apower : %.10f \n", j, aPower);
        printf("\tSource %d -  Vpower : %.10f \n", j, vPower);

    }

}

void Sources::updateSourceMixingParameter(const VectorMatrixXcd & A_global){
    int sum = 0;
    for (int j = 0; j < sources(); j++) {
        if (m_sources[j].isInst()) {
            m_sources[j].A(0) = A_global(0).block(0, sum, channels(), m_sources[j].rank());
        }
        else {
            for (int f = 0; f < bins(); f++) {
                m_sources[j].A(f) =
					A_global(f).block(0, sum, channels(), m_sources[j].rank());
            }
        }
        sum += m_sources[j].rank();
    }
}

void Sources::updateSpectralPower(NaturalStatistics &stats) {
	int J = sources();

    vector<int> beginId;
    vector<int> endId;
    for (int j = 0; j < J; j++) {
        if (j == 0){
            beginId.push_back(0);
        }
        else {
            beginId.push_back(endId[j - 1] + 1);  
        }
        endId.push_back(beginId[j] + m_sources[j].rank() - 1);  
    }
    /* 
    With openMP enabled :
    - Time consumption is reduced (divided by 2 for Example 2) => observed in practice
    But ...
    - To be tested : Memory usage increase because of NxF matrix Xi allocation on each thread ?
    */
    //TODO: Check ram consuption if omp enabled
    #pragma omp parallel for

    for (int j = 0; j < J; j++) {
        // Compute Xi(f,n): Eq. 29
        ArrayXXd Xi = ArrayXXd::Zero(m_bins, m_frames);
		for (int f = 0; f < m_bins; f++) {
			for (int n = 0; n < m_frames; n++) {
            
                for (int r = beginId[j]; r <= endId[j]; r++) {
                    Xi(f, n) += stats.hatRs(n , f)(r, r).real();
                }
            }
        }
        Xi /= m_sources[j].rank();

        // Update spectral parameters
        m_sources[j].updateSpectralPower(Xi);
    }
}

void Sources::spectralPowerSmoothing() {
	   // Optional temporal and frequency smoothing
	   for (int j = 0; j < sources(); j++) {
	       if(m_sources[j].wiener_c1()!=0 || m_sources[j].wiener_c2()!=0){
	           m_sources[j].spectralPowerSmoothing();
	       }
	   }
}

void Sources::write(const string dirname , int samplingRate, vector<fasst::Audio> &output) {
    int J = sources();
	for (int j = 0; j < J; j++) {
		output[j].write(dirname + m_sources[j].name() + ".wav", samplingRate);
	}
}

Eigen::MatrixXcd Sources::computeW(int n, int f, int j) const {
	// Compute Sigma_x inverse with annealing term
	int I = channels();
	int J = sources();

	MatrixXcd Sigma_x = MatrixXcd::Identity(I, I)*m_noise_end(f);
	MatrixXcd currentSigmaY;
	for (int jj = 0; jj < J; jj++) {
		if (j == jj) {
			currentSigmaY = m_sources[jj].compSigmaY(f, n);
			Sigma_x += currentSigmaY;
		}
		else {
			Sigma_x += m_sources[jj].compSigmaY(f, n);
		}
	}
	return Source::WienerThresholding(currentSigmaY * Sigma_x.inverse(), m_sources[j].wiener_qd());
}


bool Sources::equal(const Sources & ss, double margin) const {
    if ((*this).bins() != ss.bins() ||
        (*this).frames() != ss.frames() ||
        (*this).channels() != ss.channels() ||
        (*this).sources() != ss.sources()) {
        cout << "Dimension MISMATCH" << endl;
        return false;
    }
    else {
        bool isOk = true;
        for (int j = 0; j < (*this).sources(); j++) {
            cout << "Compare content of source " << to_string(j) << endl;
            isOk = isOk && (*this)[j].equal(ss[j], margin);
        }
        return isOk;

    }
}
}
