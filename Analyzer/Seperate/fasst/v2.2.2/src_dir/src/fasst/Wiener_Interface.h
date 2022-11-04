// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_WIENER_INTERFACE_H
#define FASST_WIENER_INTERFACE_H

//#include "matrix.h"  // Eigen Matrix typedef


/*!
This class is a minimal interface for Wiener Filtering purpose
*/

namespace fasst {
    class Wiener_Interface {
    public:
        
		/*!
		This method computes the Wiener gain for a time-frequency point (n,f) and a source j
		\param n The time frame index
		\param f The frequency bin index
		\param j The source index
		\return The I x I Wiener gain matrix with I the number of channels
		*/
        virtual Eigen::MatrixXcd computeW(int n, int f, int j) const = 0;

        /*!
        This method returns the number of time frames
        */
        virtual int frames() const = 0;

        /*!
        This method returns the number of frequency bins
        */
        virtual int bins() const = 0;

        /*!
        This method returns the number of sources
        */
        virtual int sources() const = 0;

        /*!
        This method returns the number of channels
        */
        virtual int channels() const = 0;

		/*!
		This method returns the filtered source name at index j
		*/
		virtual std::string name(int j) const = 0;
    };
}
#endif