// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_PARAMETER_H
#define FASST_PARAMETER_H

#include "tinyxml2.h"
#include <string>

namespace fasst {

/*!
 This class is an abstract base class for every source parameters. It has an
 attribute corresponding to its degree of adaptability
 */
class Parameter {
public:

  /*!
   \return `true` is the degree of adaptability is free, `false` if it is fixed.
   */
  inline bool isFree() const { return m_adaptability == "free"; }

protected:
    /*!
    Source adaptability (free or fixed)
    */
  std::string m_adaptability;
};
}

#endif
