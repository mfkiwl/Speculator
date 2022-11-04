// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_SOURCE_H
#define FASST_SOURCE_H

#include "MixingParameter.h"
#include "SpectralPower.h"
#include "Comparable.h"
#include "tinyxml2.h"

namespace fasst {
/*!
 This class represents a source with all of its 9 parameters: one
 MixingParameter and eight NonNegMatrix. In addition, it has an attribute for V
 which models the entire spectral power of the source and is stored in a
 `Eigen::MatrixXd` object which can be seen as a \f$F \times N\f$-matrix, and an
 attribute for R which models the spatial covariance matrix and is stored in a
 `VectorMatrixXcd` object which can be seen as a \f$F\f$-vector of \f$I \times
 I\f$-complex-matrices.
 */
class Source : public Comparable<Source> {
public:
  /*!
   The main constructor of the class. It loads each parameter from an XML node,
   and then computes V and R from the parameters.
   \param xmlNode The XML node containing the source parameters.
   \param index of the source
   */
	Source(const tinyxml2::XMLElement* xmlNode,int index);

  /*!
   This method is called during the M-step of the EM algorithm. It calls the
   update method either on both SpectralPower, or only on the excitation part if
   the filter part is null. Then the method computes V from the new value of the
   parameters.
   */
  void updateSpectralPower(const Eigen::ArrayXXd &Xi);

  /*!
   \return `true` is the mixing type of the source is instantaneous, `false` if
   it is convolutive.
   */
  inline bool isInst() const { return m_A.isInst(); }
  /*!
   \return `true` is the mixing type of the source is convolutive, `false` if it
   is instantaneous.
   */
  inline bool isConv() const { return m_A.isConv(); }

  /*!
   \return the rank of the spatial covariance of the source.
   */
  inline int rank() const { return m_A.rank(); }

  /*!
   \return the name of the source.
   */
  inline std::string name() const { return m_name; }

    /*!
   \return Wiener parameter 'a' of the source.
   */
  inline double wiener_qa() const { return m_wiener_qa; }
  /*!
  \return Wiener parameter 'b' of the source.
  */
  inline Eigen::MatrixXcd wiener_b() const { return m_wiener_b; }
  /*!
  \return Wiener parameter 'c1' of the source.
  */
  inline int wiener_c1() const { return m_wiener_c1; }
  /*!
  \return Wiener parameter 'c2' of the source.
  */
  inline int wiener_c2() const { return m_wiener_c2; }
  /*!
  \return Wiener parameter 'd' of the source.
  */
  inline double wiener_qd() const { return m_wiener_qd; }

  /*!
   \return the mixing parameter of the source.
   */
  inline const MixingParameter &A() const { return m_A; }

  /*!
   This method is used to get the value of the mixing parameter of the source at
   a given frequency bin.
   \param bin the frequency bin index
   \return the mixing parameter of the source at the given frequency bin.
   */
  inline const Eigen::MatrixXcd &A(int bin) const { return m_A(bin); }

  /*!
   This method is used to set the value of the mixing parameter of the source at
   a given frequency bin.
   \param bin the frequency bin index
   \return the mixing parameter of the source at the given frequency bin.
   */
  inline Eigen::MatrixXcd &A(int bin) { return m_A(bin); }

  /*!
   This method is used to get the value of the spectral power `V` in one TF
   point.
   \param bin the frequency bin index
   \param frame the time frame index
   \return the value of `V` corresponding to the indexes.
   */
  inline double V(int bin, int frame) const { return m_V(bin, frame); }

  /*!
  This method is used to get the spectral power `V` 
  \return V
  */
  inline const Eigen::ArrayXXd &V() const { return m_V; }

  /*!
   This method is used to get the covariance matrix `Sigma_y` in one time-frequency
   bin.
   */
  Eigen::MatrixXcd compSigmaY(int bin, int frame) const;

  /*!
   This method computes V. It is an implementation of \ref eq "Eq. 9". It is
   called when the object is constructed and after each M-step.
   */
  void compV();

  /*!
   This method applied a 2D filter on the spectral power parameter (m_V). It is called optionally before Wiener Filtering.
   */
  void spectralPowerSmoothing();

  /*!
   This method apply Wiener thresholding on matrix m which corresponds for one (f,n) to Sigma_y * Sigma_x-1
   */
  static Eigen::MatrixXcd WienerThresholding(const Eigen::MatrixXcd &m, const double & wiener_qd);

  /*!
   \return the number of frequency bins
   */
  inline int bins() const { return static_cast<int>(m_V.rows()); }

  /*!
   \return the number of time frames
   */
  inline int frames() const { return static_cast<int>(m_V.cols()); }
  
  /*!
  \return true if excitation only
  */
  inline bool excitationOnly() const { return m_excitationOnly; }
  /*!
  \return Spectral Power matrix "ex"
  */
  inline const SpectralPower& getSPEx() const { return m_ex; };

  /*!
  \return Spectral Power matrix "ft"
  */
  inline const SpectralPower& getSPFt() const { return m_ft; };

  /*!
  This method is an equality test between the current Source object and the one passed in parameter.
  Source objects are equals if :
  - their mixing parameter (A matrix) are equals (according to a margin)
  - their exitation spectral power are equals (according to a margin)
  - their filter spectral power are equals (according to a margin)
  \param s : Source object to be compared
  \param margin : relative margin tolerance expressed in %
  \return true if Sources are equals, false otherwise
  */
  bool equal(const Source & s, double margin) const;

private:
  std::string m_name;
  MixingParameter m_A;
  SpectralPower m_ex, m_ft;

  bool m_excitationOnly;

  int m_bins;
  
  double m_wiener_qa;
  MatrixXcd m_wiener_b;
  int m_wiener_c1;
  int m_wiener_c2;
  double m_wiener_qd;
 
  Eigen::ArrayXXd m_V;
};
}

#endif
