#ifndef FASST_COMPARABLE_H
#define FASST_COMPARABLE_H

#include "typedefs.h"
#include "EigenTypeComparator.h"
#include <iostream>

namespace fasst {

    /*!
     This class is an interface, all sub class must implement the equal method.
	 It's derived to EigenTypeComparator which gives some methods to compare Eigen types.
     */
	template <class T>
    class Comparable : public EigenTypeComparator {
    public:
		/*!
		This pure virtual method is an equality test between the calling object and
		the one passed in parameter. Margin is a parameter to control the error margin
		*/
		virtual bool equal(const T& m, double margin) const = 0;
    };
}

#endif
