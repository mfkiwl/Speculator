// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "NaturalStatistics.h"
#include "MixCovMatrix.h"
#include "Sources.h"
#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;
using namespace std;

namespace fasst {
	NaturalStatistics::NaturalStatistics(const Sources &sources, int nIters) : m_nIters(nIters) {
		m_frames = sources.frames();
		m_bins = sources.bins();
		m_channels = sources.channels();
		int R = 0;

		for (int j = 0; j < sources.sources(); j++) {
			R += sources[j].rank();
		}
		m_hatRxs = ArrayMatrixXcd(m_frames, m_bins);
		m_hatRs = ArrayMatrixXcd(m_frames, m_bins);

		//
		for (int n = 0; n < m_frames; n++) {
			for (int f = 0; f < m_bins; f++) {
				m_hatRs(n , f) = MatrixXcd(R, R);
				m_hatRxs(n , f) = MatrixXcd(m_channels, R);
			}
		}

		// Indices
		int current_index = 0;
		for (int j = 0; j < sources.sources(); j++) {
			vector<int> ind_j(sources[j].rank());
			for (int i = 0; i < sources[j].rank(); i++) {
				ind_j[i] = current_index;
				current_index++;
			}

			// Here we concatenate the C (or I) subset of indices with the j-th source's
			// indices
			if (sources[j].A().isFree() && sources[j].isInst()) {
				m_ind_I.insert(m_ind_I.end(), ind_j.begin(), ind_j.end());
			}
			else {
				m_ind_Icomp.insert(m_ind_Icomp.end(), ind_j.begin(), ind_j.end());
			}
			if (sources[j].A().isFree() && sources[j].isConv()) {
				m_ind_C.insert(m_ind_C.end(), ind_j.begin(), ind_j.end());
			}
			else {
				m_ind_Ccomp.insert(m_ind_Ccomp.end(), ind_j.begin(), ind_j.end());
			}
		}
		m_sumEq26 = MatrixXcd::Zero(m_channels, m_ind_C.size());
		m_sum_hat_Rs_C = MatrixXcd::Zero(m_ind_C.size(), m_ind_C.size());
		m_rhs26 = MatrixXcd(m_channels, m_ind_C.size());
		m_A_Ccomp = MatrixXcd(m_channels, m_ind_Ccomp.size());
		m_hat_Rxs_C = MatrixXcd(m_channels, m_ind_C.size());
		m_hat_Rs_Ccomp = MatrixXcd(m_ind_Ccomp.size(), m_ind_C.size());
		m_hat_Rs_Icomp = MatrixXcd(m_ind_Icomp.size(), m_ind_I.size());
		m_sumEq27 = MatrixXcd::Zero(m_channels, m_ind_I.size());
		m_sum_hat_Rs_I = MatrixXcd::Zero(m_ind_I.size(), m_ind_I.size());
		m_A_Icomp = MatrixXcd(m_channels, m_ind_Icomp.size());
		m_hat_Rxs_I = MatrixXcd(m_channels, m_ind_I.size());
		m_rhs27 = MatrixXd(m_channels, m_ind_I.size());

		// Compute A from each source mixing parameter
		m_A_global = VectorMatrixXcd(m_bins);
		for (int f = 0; f < m_bins; f++) {
			m_A_global(f) = MatrixXcd::Zero(m_channels, R);
			int sum = 0;
			for (int j = 0; j < sources.sources(); j++) {
				if (sources[j].isInst()) {
					m_A_global(f).block(0, sum, m_channels, sources[j].rank()) =
						sources[j].A(0);
				}
				else {
					m_A_global(f).block(0, sum, m_channels, sources[j].rank()) =
						sources[j].A(f);
				}
				sum += sources[j].rank();
			}
		}
	}

	void NaturalStatistics::EStep(const Sources &sources,
		const MixCovMatrix &hatRx,
		bool noSimAnn,
		int currentIter) {

		int F = hatRx.bins();
		int N = hatRx.frames();
		int I = hatRx.channels();
		int R = static_cast<int>(m_A_global(0).cols());
		int J = sources.sources();
		double log_like = 0;

#pragma omp parallel for reduction(+ : log_like)
		for (int f = 0; f < F; f++) {
			double sigma_f;
			if (noSimAnn) {
				sigma_f = sqrt(sources.getNoiseEnd(f));
			}
			else {
				sigma_f = (sqrt(sources.getNoiseBeg(f)) * (m_nIters - currentIter - 1) +
					sqrt(sources.getNoiseEnd(f)) * (currentIter + 1)) / m_nIters;
			}
			MatrixXd Sigma_b = MatrixXd::Identity(m_channels, m_channels) * sigma_f * sigma_f;
			MatrixXcd tmpMat(R, I);
			MatrixXcd Sigma_x(I, I);
			MatrixXcd Sigma_x_inverse(I, I);
			MatrixXd Sigma_s(R, R);
			Sigma_s.setZero();
			MatrixXcd Omega_s(R, I);
			PartialPivLU<MatrixXcd> Sigma_x_lu = PartialPivLU<MatrixXcd>(static_cast<Eigen::Index>(I));
			for (int n = 0; n < N; n++) {
				// Eq. 25 
				int sum = 0;

				for (int j = 0; j < J; j++) {
					Sigma_s.diagonal().segment(sum, sources[j].rank()).array() = sources[j].V(f, n);
					sum += sources[j].rank();
				}

				tmpMat.noalias() = Sigma_s*m_A_global(f).adjoint(); 

				// Eq. 24
				Sigma_x.noalias() = m_A_global(f)*tmpMat + Sigma_b;
				
				// Sigma_x inverse computation
				Sigma_x_lu.compute(Sigma_x);
				Sigma_x_inverse.noalias() = Sigma_x_lu.inverse();				

				// Eq. 23
				Omega_s.noalias() = tmpMat * Sigma_x_inverse;

				// Eq. 21
				m_hatRxs(n, f).noalias() = hatRx(f, n) * Omega_s.adjoint();


				// Eq. 22
				m_hatRs(n, f).noalias() = (Omega_s*(m_hatRxs(n, f) - tmpMat.adjoint())) + Sigma_s;

				// Log-likelihood: Eq. 16
				double det = Sigma_x_lu.determinant().real();
				log_like -= (Sigma_x_inverse * hatRx(f, n)).real().trace() +
					log(det * M_PI);

			}
		}
		log_like /= (F * N);
		m_logLikelihood.push_back(log_like);
	}

	void NaturalStatistics::updateMixingParameter() {

		// ------------------ Eq. 26 ------------------	

		for (int f = 0; f < m_bins; f++) {
			for (int i = 0; i < static_cast<int>(m_ind_Ccomp.size()); i++) {
				m_A_Ccomp.col(i) = m_A_global(f).col(m_ind_Ccomp[i]);
			}

			m_sumEq26.setZero();
			m_sum_hat_Rs_C.setZero();
			for (int n = 0; n < m_frames; n++) {

				for (int i = 0; i < static_cast<int>(m_ind_C.size()); i++) {
					m_hat_Rxs_C.col(i) = m_hatRxs(n, f).col(m_ind_C[i]);
				}

				for (int j = 0; j < static_cast<int>(m_ind_C.size()); j++) {
					for (int ii = 0; ii < static_cast<int>(m_ind_C.size()); ii++) { // add
						m_sum_hat_Rs_C(ii, j) += m_hatRs(n, f)(m_ind_C[ii], m_ind_C[j]);
					}
					for (int i = 0; i < static_cast<int>(m_ind_Ccomp.size()); i++) {
						m_hat_Rs_Ccomp(i, j) = m_hatRs(n, f)(m_ind_Ccomp[i], m_ind_C[j]);
					}
				}
				m_sumEq26 += m_hat_Rxs_C - m_A_Ccomp * m_hat_Rs_Ccomp;
			}

			m_rhs26 = m_sumEq26 * m_sum_hat_Rs_C.inverse();
			for (int i = 0; i < static_cast<int>(m_ind_C.size()); i++) {
				m_A_global(f).col(m_ind_C[i]) = m_rhs26.col(i);
			}
		} // end for f

		  // ------------------ Eq. 27 ------------------

		m_sumEq27.setZero();
		m_sum_hat_Rs_I.setZero();

		for (int f = 0; f < m_bins; f++) {

			for (int i = 0; i < static_cast<int>(m_ind_Icomp.size()); i++) {
				m_A_Icomp.col(i) = m_A_global(f).col(m_ind_Icomp[i]);
			}

			for (int n = 0; n < m_frames; n++) {
				for (int i = 0; i < static_cast<int>(m_ind_I.size()); i++) {
					m_hat_Rxs_I.col(i) = m_hatRxs(n, f).col(m_ind_I[i]);
				}

				for (int j = 0; j < static_cast<int>(m_ind_I.size()); j++) {
					for (int ii = 0; ii < static_cast<int>(m_ind_I.size()); ii++) {
						m_sum_hat_Rs_I(ii, j) += m_hatRs(n, f)(m_ind_I[ii], m_ind_I[j]);
					}

					for (int i = 0; i < static_cast<int>(m_ind_Icomp.size()); i++) {
						m_hat_Rs_Icomp(i, j) = m_hatRs(n, f)(m_ind_Icomp[i], m_ind_I[j]);
					}
				}
				m_sumEq27 += m_hat_Rxs_I - m_A_Icomp * m_hat_Rs_Icomp;
			}
		}//end for f


		m_rhs27 = m_sumEq27.real() * m_sum_hat_Rs_I.real().inverse();
		for (int f = 0; f < m_bins; f++) {
			for (int i = 0; i < static_cast<int>(m_ind_I.size()); i++) {
				m_A_global(f).col(m_ind_I[i]).real() = m_rhs27.col(i);
			}
		}
	}
}