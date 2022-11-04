// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_MIXINGPARAMETER_H
#define FASST_MIXINGPARAMETER_H

#include "typedefs.h"
#include "Parameter.h"
#include <complex>
#include "Comparable.h"

namespace fasst {

/*!
 This class represents the mixing parameter of a source, which models its
 spatial covariance. Data is stored in a `VectorMatrixXcd` object, which can be
 seen as a \f$F\f$-vector of \f$I \times R\f$-matrices. \f$F\f$ is the number of
 frequency bins, \f$I\f$ is the number of audio channels and \f$R\f$ is the rank
 of the spatial covariance.

 \remark Three cases are possible:
   1. The mixing type is instantaneous and the data is real and constant over
 frequency (_ie._ `size() == 1`).
   2. The mixing type is convolutive and the data is real and
 frequency-dependent.
   3. The mixing type is convolutive and the data is complex and
 frequency-dependent.

  If this implementation allows another combination, it's a bug, not a feature.

 \todo Document the format of the data in the XML file: [a
 link](http://stackoverflow.com/questions/16898691/pythonic-way-to-print-a-multidimensional-complex-numpy-array-to-a-string)
     */
class MixingParameter : public Parameter, public VectorMatrixXcd, public Comparable<MixingParameter> {
public:
  /*!
   The main constructor of the class loads data and other attributes from an XML
   element. If the input is not consistent, this constructor will throw a
   `runtime_error` exception
   \param xmlEl An XML element
   */
	MixingParameter(const tinyxml2::XMLElement* xmlEl);

	/*!
	This method is used to get the complex value of the mixing parameter at a given frequency index
	(freqId), channel index (channelId) and rank index (rankId).
	Note : if mixing type is instantaneous, freqId is ignored
	\param freqId Frequency bin index
	\param channelId Channel index
	\param rankId Rank index
	\return A complex value
	*/
	std::complex<double> getVal(int freqId, int channelId, int rankId) const ;
  
  /*!
   \return `true` is the mixing type is instantaneous, `false` if it is
   convolutive.
   */
  inline bool isInst() const { return m_mixingType == "inst"; }

  /*!
   \return `true` is the mixing type is convolutive, `false` if it is
   instantaneous.
   */
  inline bool isConv() const { return m_mixingType == "conv"; }

  /*!
   \return the rank of the spatial covariance.
   */
  inline int rank() const { return static_cast<int>((*this)(0).cols()); }

  /*!
  This method is an equality test between the current MixingParameter object and the one passed in parameter.
  MixingParameter objects are equals if their content (A matrix) is the same (with a relative margin) and their types are identicals
  \param m : MixingParameter object to be compared
  \param margin : relative margin tolerance expressed in %
  \return true if MixingParameter are equals, false otherwise
  */
  bool equal(const MixingParameter & m, double margin) const ;

private:
  std::string m_mixingType;
};
}

#endif
