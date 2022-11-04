// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_SOURCES_H
#define FASST_SOURCES_H

#include "Source.h"
#include "STFTRepr.h"
#include "Audio.h"
#include "MixCovMatrix.h"
#include "Comparable.h"
#include "Wiener_Interface.h"

namespace fasst {
class NaturalStatistics;

/*!
 This class represents a set of sources. In addition, it has an attribute for A
 which models the global mixing parameter containing each source. This parameter
 is stored as a `VectorMatrixXcd` which can be seen as a \f$F\f$-vector of \f$I
 \times R\f$-complex-matrices, where \f$F\f$ is the number of frequency bins,
 \f$I\f$ is the number of audio channels and \f$R\f$ is the sum of the spatial
 covariance rank of each source.
 This class implements Wiener_Interface.
 */
class Sources : Comparable<Sources> ,  public Wiener_Interface {
public:
  
	
	/*!
   The main constructor of the class load sources from a list of `<source>` XML
   elements and then compute A from them.
   \param xmlNode node of `<source>` XML elements
   */
  Sources(const tinyxml2::XMLElement* xmlNode);

  /*!
  This method is used to normalize sources' model parameter A as done in FASST v1
  \param hatRx
  */
  void preEmNormalization(const MixCovMatrix &hatRx);


  /*!
  This method is called before xml source replacement in order to
  update each source mixing parameter from the global mixing parameter
  */
  void updateSourceMixingParameter(const VectorMatrixXcd & A_global);

  /*!
   This method does two things:
   1. It computes Xi with natural statistics, which corresponds to \ref eq "Eq.
   29"
   2. It calls the Source::updateSpectralPower on each source.
   */
  void updateSpectralPower(NaturalStatistics &stats);


  /*!
  This method write out on disk each source
  \param dirname output directory name where sources are saved
  \param samplingRate sampling frequency
  \param output vector of audio to save
  */
  void write(const std::string dirname,int samplingRate, std::vector<fasst::Audio> &output);

  /*!
  This method computes the Wiener gain for a time-frequency point (n,f) and a source j
  \param n The time frame index
  \param f The frequency bin index
  \param j The source index
  \return The I x I Wiener gain matrix with I the number of channels
  */
  Eigen::MatrixXcd computeW(int n, int f, int j) const override;

  /*!
  This method returns the name of the source at the h-th index
  \param j the source index
  \return the source name at the index
  */
  inline std::string name(int j) const override { return m_sources[j].name(); };

  /*!
  This method set the noise begin vector, otherwise it is considered to be zero
  \param the noise vector to be set
  */
  void setNoiseBeg(const Eigen::VectorXd &noise) { m_noise_beging = noise; };

  /*!
  This method return the noise begin at a frequency index f
  \param the frequency index
  \return the value of the noise begin at index f
  */
  inline double getNoiseBeg(int f) const { return m_noise_beging[f]; }

  /*!
  This method set the noise end vector, otherwise it is considered to be zero
  \param the noise vector to be set
  */
  void setNoiseEnd(const Eigen::VectorXd &noise) { m_noise_end = noise; };

  /*!
  This method return the noise end at a frequency index f
  \param the frequency index
  \return the value of the noise end at index f
  */
  inline double getNoiseEnd(int f) const { return m_noise_end[f]; }

  /*!
  This method apply a spectral power smoothing on each source
  */
  void spectralPowerSmoothing();


  /*!
   This overloaded []-operator gives acces to an individual source.
   \param i the source index
   \return the source corresponding to the index
   */
  inline Source &operator[](int i) { return m_sources[i]; }

  /*!
  This overloaded []-operator gives acces to an individual source (const version).
  \param i the source index
  \return the source corresponding to the index
  */
  inline const Source &operator[](int i) const { return m_sources[i]; }
  
  /*!
   \return the number of sources
   */
  inline int sources() const override { return static_cast<int>(m_sources.size()); }

  /*!
   \return the number of frequency bins
   */
  inline int bins() const override { return m_bins; }

  /*!
   \return the number of time frames
   */
  inline int frames() const override { return m_frames; }

  /*!
   \return the number of channels
   */
  inline int channels() const override { return m_channels; }

  /*!
  This method is an equality test between the current Sources object and the one passed in parameter.
  Sources objects are equals if :
  - their number of frequency bins are equals
  - their number of time frames are equals
  - their number of channels are equals
  - the source number is equal
  - all sources composing the Sources object are equals (according to a margin)
  \param ss : Sources object to be compared
  \param margin : relative margin tolerance expressed in %
  \return true if Sources are equals, false otherwise
  */
  bool equal(const Sources & ss, double margin) const;

private:
  std::vector<Source> m_sources;
  Eigen::VectorXd m_noise_end;
  Eigen::VectorXd m_noise_beging;
  int m_bins, m_frames, m_channels;
};
}

#endif
