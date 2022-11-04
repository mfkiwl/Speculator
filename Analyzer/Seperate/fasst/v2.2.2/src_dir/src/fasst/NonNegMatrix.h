// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_NONNEGMATRIX_H
#define FASST_NONNEGMATRIX_H

#include "Parameter.h"
#include <Eigen/Core>
#include "Comparable.h"

namespace fasst {

/*!
 This class represents a nonnegative matrix which is a part of the spectral
 power of a source. It is inherited by the W, UG and H classes. All of its
 subclasses are able to update themselves during the M-step of the EM algorithm.

 In case the matrix is not defined in XML, the m_eye flag is set to true and the matrix behaves like an identity matrix. We implement multiplication and transpose operations to avoid wasting time doing operations on identity matrices.
     */
class NonNegMatrix : public Parameter, public Eigen::MatrixXd, public Comparable<NonNegMatrix>{
public:
  /*!
   The main constructor of the class loads data and other attributes from an XML
   element.
   \param xmlEl An XML element
   */
  NonNegMatrix(const tinyxml2::XMLElement* xmlEl);

  /*!
   Kind of copy constructor to create NonNegMatrix from the result of an operation on MatrixXd.
   */
  NonNegMatrix(const Eigen::MatrixXd &m) : Eigen::MatrixXd(m), m_eye(false) {}
  
  /*!
  This method is used to get the entry of a non negative matrix at given row and column indices
  \param rowId Row index
  \param colId Column index
  \return the value at (rowId,colId)
  */
  inline double getVal(int rowId, int colId) const { return this->array()(rowId, colId); };

  /*!
  Non-negative matrix product
  \return the non-negative resulting from the product
  */
  inline NonNegMatrix operator*(const NonNegMatrix &rhs) const {
    if (isEye()) {
      return rhs;
    } else if (rhs.isEye()) {
      return *this;
    } else {
      return NonNegMatrix(Eigen::MatrixXd::operator*(rhs));
    }
  }

  /*!
  Non-negative matrix transposition
  \return the transposed non-negative matrix
  */
  inline NonNegMatrix transpose() const {
    if (isEye()) {
      return *this;
    } else {
      return NonNegMatrix(Eigen::MatrixXd::transpose());
    }
  }

  /*!
  Test if the non-negative matrix is the identity matrix
  \return true if identity matrix
  */
  inline bool isEye() const { return m_eye; }

  /*!
  This method is an equality test between the current NonNegMatrix object and the one passed in parameter.
  NonNegMatrix objects are equals if their content are equals (according to a margin).
  \param m : NonNegMatrix object to be compared
  \param margin : relative margin tolerance expressed in %
  \return true if NonNegMatrix are equals, false otherwise
  */
  bool equal(const NonNegMatrix & m, double margin) const;

private:
  bool m_eye;
};

/*!
 This class represents a W parameter of a source.
 */
class W : public NonNegMatrix {
public:
  /*!
   The main constructor of the class just calls NonNegMatrix::NonNegMatrix
   constructor
   */
	W(const tinyxml2::XMLElement* xmlEl) : NonNegMatrix(xmlEl) {}
    
  /*!
   This method updates the data during the EM algorithm. It is a implementation
   of \ref eq "Eq. 30" optimized for W and for the case where the source is
   excitation-only (_ie._ `E == ones(F, N)`).
   */
  void update(const Eigen::ArrayXXd &Xi, const NonNegMatrix &D);

  /*!
   This method updates the data during the EM algorithm. It is a implementation
   of \ref eq "Eq. 30" optimized for W.
   */
  void update(const Eigen::ArrayXXd &Xi, const NonNegMatrix &D,
              const Eigen::ArrayXXd &E);
};

/*!
 This class represents either a U or a G parameter of a source.
 */
class UG : public NonNegMatrix {
public:
  /*!
   The main constructor of the class just calls NonNegMatrix::NonNegMatrix
   constructor
   */
	UG(const tinyxml2::XMLElement* xmlEl) : NonNegMatrix(xmlEl) {}

  /*!
   This method updates the data during the EM algorithm. It is an implementation
   of \ref eq "Eq. 30" optimized for U and G and for the case where the source
   is
   excitation-only (_ie._ `E == ones(F, N)`).
   */
  void update(const Eigen::ArrayXXd &Xi, const NonNegMatrix &B,
              const NonNegMatrix &D);

  /*!
   This method updates the data during the EM algorithm. It is an implementation
   of \ref eq "Eq. 30" optimized for U and G.
   */
  void update(const Eigen::ArrayXXd &Xi, const NonNegMatrix &B,
              const NonNegMatrix &D, const Eigen::ArrayXXd &E);
};

/*!
 This class represents an H parameter of a source.
 */
class H : public NonNegMatrix {
public:
  /*!
   The main constructor of the class just calls NonNegMatrix::NonNegMatrix
   constructor
   */
	H(const tinyxml2::XMLElement* xmlEl) : NonNegMatrix(xmlEl) {}

  /*!
   This method updates the data during the EM algorithm. It is an implementation
   of \ref eq "Eq. 30" optimized for H and for the case where the source is
   excitation-only (_ie._ `E == ones(F, N)`).
   */
  void update(const Eigen::ArrayXXd &Xi, const NonNegMatrix &B);

  /*!
   This method updates the data during the EM algorithm. It is an implementation
   of \ref eq "Eq. 30" optimized for H.
   */
  void update(const Eigen::ArrayXXd &Xi, const NonNegMatrix &B,
              const Eigen::ArrayXXd &E);
};
}

#endif
