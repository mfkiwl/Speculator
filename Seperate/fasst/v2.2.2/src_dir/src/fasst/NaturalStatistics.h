// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_NATURALSTATISTICS
#define FASST_NATURALSTATISTICS

#include "typedefs.h"

namespace fasst {
class Sources;
class MixCovMatrix;

/*!
 This class contains the natural sufficient statistics of the EM algorithm. It
 is used for the E-step of the algorithm and for the computation of the
 log-likelihood.
 */
class NaturalStatistics {
public:
	/*!
	The main constructor of the class only allocate memory to private member of the class
	\param sources
	\param total number of iterations
	*/
	NaturalStatistics(const Sources &sources, int nIters);
	
  /*!
   Computation for the E-step of the algorithm (\ref eq "Eq. 21 to 25") 
   and the computation of the log-likelihood
   (\ref eq "Eq. 16").
   \param sources the sources to be updated
   \param hatRx the mixture covariance matrix
   \param noSimAnn boolean to switch off the simulated annealing
   \param currentIter the current iteration of the E-M algorithm
   */
  void EStep(const Sources &sources, const MixCovMatrix &hatRx, bool noSimAnn, int currentIter);

  /*!
  This method is an implementation of \ref eq "Eq. 26" and \ref eq "Eq. 27". It
  computes an update of A (global) with natural statistics.
  */
  void updateMixingParameter();

  /*!
   This function is used to get the value of the log-likelihood.
   \return the value of the log-likelihood.
   */
  inline double logLikelihood() const { return m_logLikelihood.back(); }

  /*!
  This function is used to get the value of the log-likelihood improvement between two iterations.
  \return the value of the log-likelihood improvement.
  */
  inline double logLikelihoodImprovement() const {
	  return (m_logLikelihood.back() - m_logLikelihood[m_logLikelihood.size() - 2]);
  }
  
  /*!
   This method is used to get the natural statistic \f$\hat{R_{s}}\f$ at a given
   TF point.
   \param bin the frequency bin index
   \param frame the time frame index
   \return the value of \f$\hat{R_{s}}\f$ corresponding to the indexes.
   */
  inline const Eigen::MatrixXcd &hatRs(int bin, int frame) const {
    return m_hatRs(bin, frame);
  }

  // Accessors (read only)
  /*!
  This method is used to get the value of the global mixing parameter at a
  given frequency bin.
  \param bin the frequency bin index
  \return the \f$I \times R\f$-matrix corresponding to the frequency bin index
  */
  inline const Eigen::MatrixXcd &A_global(int bin) const { return m_A_global(bin); }

  /*!
  This method is used to get the global mixing parameter.
  \return the \f$F \times \f$I \times R\f$-matrix
  */
  inline const VectorMatrixXcd &A_global() const { return m_A_global; }
  

private:
   VectorMatrixXcd m_A_global;
   int m_frames, m_bins, m_channels;
   int m_nIters; // total number of iteration
  
				 // Used for E-Step
  ArrayMatrixXcd m_hatRxs, m_hatRs;
  std::vector<double> m_logLikelihood;

  // Used for M-Step
  
  // used in updateMixingParameter
  std::vector<int> m_ind_C, m_ind_Ccomp, m_ind_I, m_ind_Icomp; 
  Eigen::MatrixXcd m_sumEq26, m_rhs26, m_sum_hat_Rs_C, m_A_Ccomp, m_hat_Rxs_C, m_hat_Rs_Ccomp, m_hat_Rs_Icomp, m_sumEq27;
  Eigen::MatrixXd m_rhs27;
  Eigen::MatrixXcd m_sum_hat_Rs_I, m_A_Icomp, m_hat_Rxs_I;

 
};
}

#endif
