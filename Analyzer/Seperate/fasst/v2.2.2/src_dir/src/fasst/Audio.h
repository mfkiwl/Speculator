// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_AUDIO_H
#define FASST_AUDIO_H

#include <Eigen/Core>
#include "Comparable.h"

namespace fasst {

/*!
 This class contains audio data. The audio data is stored in an
 `Eigen::ArrayXXd` object, which can be seen as an \f$I \times N\f$-array where \f$N\f$ is the
number of time frames and \f$I\f$ is the number of audio channels (rows of the array are audio channels and columns of
 the array are audio samples.)
 */
class Audio : public Eigen::ArrayXXd, public Comparable<Audio> {
public:
  /*!
   The main constructor of the class reads audio data from a WAV file. Please
   note that if the file is not readable, the constructor will throw a
   `runtime_error` exception.
   \param fname the name of the WAV file to be read
   */
  Audio(const char *fname);

  Audio(){};

  /*!
   This constructor build an Audio object from audio data stored in an
   `Eigen::ArrayXd` object.
   \param x the audio data to be copied
   */
  Audio(const Eigen::ArrayXXd &x) : Eigen::ArrayXXd(x) {}
  Audio(const Eigen::ArrayXXd &x, int fs) : Eigen::ArrayXXd(x), m_samplerate(fs) {}
  /*!
   This method writes the audio data to a file, at a given sample rate (sampling frequency). Please note that if the file is not writable, the method will throw a `runtime_error` exception.

   \param fname the name of the WAV file to be written
   \param samplerate the sampling frequency to be written
   */
  void write(const std::string fname, int samplerate);

  /*!
   This method writes the audio data to a file. Please note that if the file is
   not writable, the method will throw a `runtime_error` exception.

   \param fname the name of the WAV file to be written
   */
  void write(const std::string fname);

  /*!
   \return the number of audio samples
   */
  inline int samples() const { return static_cast<int>(rows()); }

  /*!
   \return the number of audio channels
   */
  inline int channels() const { return static_cast<int>(cols()); }

  /*!
   \return the sample rate (sampling frequency) in Hertz
   */
  inline int samplerate() const { return m_samplerate; }

  /*!
  This method is an equality test between the current Audio object and the one passed in parameter.
  Two Audio objects are equals if : their sample rates are equal and the audio content is the same (with a relative margin)
  \param a : audio object to be compared
  \param margin : relative margin tolerance expressed in %
  \return true if Audio objects are equals, false otherwise
  */
  bool equal(const Audio& a,double margin) const ;

private:
  int m_samplerate;
};
}

#endif
