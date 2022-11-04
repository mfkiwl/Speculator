// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_MIXCOVMATRIX_H
#define FASST_MIXCOVMATRIX_H

#include "typedefs.h"
#include "Comparable.h"
#include <complex>

namespace fasst {
class Audio;

/*!
 This class contains a mixture covariance matrix. The data is stored in an
`ArrayMatrixXcd` object, which can be seen as a \f$F \times N\f$-array of \f$I
\times I\f$-matrices.  \f$F\f$ is the number of frequency bins, \f$N\f$ is the
number of time frames and \f$I\f$ is the number of audio channels. The matrices
are Hermitian, _ie._ the diagonal elements are real and the upper triangular
part
is equal to the conjugate of the lower triangular part. We take advantage of
this property when we store the data in a binary file. The binary file format is
documented in the \ref binfileformat page.

\remark It might be possible to take advantage of the Hermitian property of the
matrices when we store the data in the object (with help of Eigen
`selfadjointView` template).

\remark In the future, it might be interesting to use the binary file format to
store other data (namely features, uncertainty).
 */
class MixCovMatrix : public Comparable<MixCovMatrix> {
public:
    /*!
    Default constructor, just assign the name of the covariance matrix, and the 
    energy signal vector and declare members
    */

    MixCovMatrix() : m_Rx_name("Rx.bin"), m_Rx_energy_name("Rx_en.bin") {};
    /*!
   The main constructor of the class computes the mixture covariance matrices of
   some audio signal (with the STFT transform) and stores it in the object.
   In addition, it computes the vector of the mean signal energy per frequency and stores it too.
   \param x a multichannel audio signal
   \param tfr_type is either STFT or ERB
   \param wlen the window length _ie._ the length (in audio samples) of one time
   frame
   \param nbin the number of frequency bins
   */
  MixCovMatrix(const Audio &x, std::string tfr_type, int wlen, int nbin, int nbinperERB = 0);

  
  MixCovMatrix(const std::string path);

  /*!
   This method reads a binary file and loads the mixture covariance
   matrices from it.
   The binary file could be :
   - The covariance matrix if mat == "Rx"
   - The mean signal energy per frequency vector if mat == "Rx_en"
   Please note that if file doesn't exist or is not
   readable, this method will throw a `runtime_error` exception.
   \param path the path to the binary file(s)
   \param mat "Rx" or "Rx_en" 
   */
  void read(std::string path, std::string mat);

  /*!
   This method writes :
   - The mixture covariance matrices to a binary file Rx.bin.
   - The mean signal energy per frequency vector in Rx_en.bin
   Please note that if the file is not writable, this method will throw a
   `runtime_error` exception.
   \param path the name of the output binary path
   */
  void write(std::string path);

  /*!
   \return the number of frequency bins
   */
  inline int bins() const { return static_cast<int>(m_Rx.rows()); }

  /*!
   \return the number of time frames
   */
  inline int frames() const { return static_cast<int>(m_Rx.cols()); }

  /*!
   \return the number of audio channels
   */
  inline int channels() const { return static_cast<int>(m_Rx(0, 0).rows()); }

  /*!
  MixCovMatrix mutator
  \return, the covariance matrix (that can be updated) at frequency bin f ant time frame n
  */
  Eigen::MatrixXcd& operator()(int const f, int const n) {
      return m_Rx(f, n);
  }

  /*!
  MixCovMatrix accessor (read only)
  \return, the covariance matrix at frequency bin f ant time frame n
  */
  Eigen::MatrixXcd operator()(int const f, int const n) const {
      return m_Rx(f, n);
  }

  /*!
  MixCovMatrix accessor (read only)
  \return, the complex element at frequency bin f, time frame n, channel i1 and i2 of the covariance matrix
  */
  std::complex<double> operator()(int const f, int const n,int const i1,int const i2) const {
      return m_Rx(f, n)(i1,i2);
  }

  /*!
  MixCovMatrix mutator
  \return, the complex element (that can be updated) at frequency bin f, time frame n, channel i1 and i2 of the covariance matrix
  */
  std::complex<double>& operator()(int const f, int const n, int const i1, int const i2) {
      return m_Rx(f, n)(i1, i2);
  }

  inline const ArrayMatrixXcd & getRx() const { return m_Rx; };
  /*!
   Mean signal energy per frequency vector accessor 
  \return, the vector (read only)
  */
  inline const Eigen::VectorXd & getRxEnergy() const  { return m_Rx_energy; };

  /*!
  Two mixCovMatrixObject are equals if both Rx and Rx_en are equals
  \param m the MixCovMatrix to be compared to
  \param margin the tolerated relative error margin
  \return true if both Rx and Rx_en are equals
  */
  bool equal(const MixCovMatrix & m, double margin) const;



private:
    /*!
    Covariance matrix representation, (FxN) array of (IxI) matrix
    */
    ArrayMatrixXcd m_Rx;
    /*!
    Covariance matrix bin file name
    */
    std::string m_Rx_name;
    /*!
    Mean signal energy per frequency vector, (Fx1) vector
    */
    Eigen::VectorXd m_Rx_energy;
    /*!
    Bin file name of the mean signal energy per frequency vector
    */
    std::string m_Rx_energy_name;

    /*!
    This method compute the mean signal energy per frequency vector
    */
    void compRxEnergy();

	/*!
	This method is an equality test between the current MixCovMatrix object and the one passed in parameter.
	It compares Rx matrices.
	Rx matrices are equals if their content are the same (with a relative margin)
	\param m : MixCovMatrix object to be compared
	\param margin : relative margin tolerance expressed in %
	\return true if Rx are equals, false otherwise
	*/
	bool equal_Rx(const MixCovMatrix & m, double margin) const;

	/*!
	This method is an equality test between the current MixCovMatrix object and the one passed in parameter.
	It compares Rx energy matrices.
	Rx_en matrices are equals if their content are the same (with a relative margin)
	\param m : MixCovMatrix object to be compared
	\param margin : relative margin tolerance expressed in %
	\return true if Rx_en are equals, false otherwise
	*/
	bool equal_Rx_en(const MixCovMatrix & m, double margin) const;
};
}

#endif
