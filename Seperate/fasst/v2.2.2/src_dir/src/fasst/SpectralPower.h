// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_SPECTRALPOWER_H
#define FASST_SPECTRALPOWER_H

#include "NonNegMatrix.h"
#include "tinyxml2.h"
#include "Comparable.h"

namespace fasst {

/*!
 This class represents either the excitation spectral power or the filter
 spectral power of a source. It has 4 NonNegMatrix objects to store W, U, G and
 H.
    */
class SpectralPower : public Comparable<SpectralPower>{
public:
  /*!
   The main constructor of the class builds the 4 NonNegMatrix attributes from
   an XML node.
   \param xmlNode An XML node
   \param suffix Should be either 'ex' or 'ft'
   */
	SpectralPower(const tinyxml2::XMLElement* xmlNode, std::string suffix);

  /*!
   This method is called during the M-step of the EM algorithm. It calls the
   update method on each NonNegMatrix which degree of adaptability is free. It
   is optimized for the case where the source doesn't have a filter spectral
   model.
   */
  void update(const Eigen::ArrayXXd &Xi);

  /*!
   This method is called during the M-step of the EM algorithm. It calls the
   update method on each NonNegMatrix whose degree of adaptability is free.
   */
  void update(const Eigen::ArrayXXd &Xi, const Eigen::ArrayXXd &E);

  /*!
   This method is an implementation of \ref eq "Eq. 12".
   \return the product of the four NonNegMatrix
   */
  inline Eigen::ArrayXXd V() const {
    return m_W * m_U * m_G * m_H;
  }

  /*!
  This method is used to get the W matrix
  \return W object
  */
  inline const W& getW() const { return m_W; };

  /*!
  This method is used to get the U object
  \return U object
  */
  inline const UG& getU() const { return m_U; };

  /*!
  This method is used to get the G object
  \return G object
  */
  inline const UG& getG() const { return m_G; };
 // inline const UG& getG() { return m_G; };
  /*!
  This method is used to get the H object
  \return H object
  */
  inline const H& getH() const { return m_H; };

  /*!
  This method is an equality test between the current SpectralPower object and the one passed in parameter.
  SpectralPower objects are equals if :
  - non negative matrix U are equals (according to a margin)
  - non negative matrix G are equals (according to a margin)
  - non negative matrix W are equals (according to a margin)
  - non negative matrix H are equals (according to a margin)
  \param m : SpectralPower object to be compared
  \param margin : relative margin tolerance expressed in %
  \return true if SpectralPower are equals, false otherwise
  */
  bool equal(const SpectralPower & m, double margin) const;


private:
  W m_W;
  UG m_U, m_G;
  H m_H;
};
}

#endif
