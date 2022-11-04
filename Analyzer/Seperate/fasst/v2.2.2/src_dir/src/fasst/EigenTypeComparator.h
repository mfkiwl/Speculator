#ifndef FASST_EIGENTYPECOMPARATOR_H
#define FASST_EIGENTYPECOMPARATOR_H
#include "typedefs.h"
using namespace Eigen;

namespace fasst {
	class EigenTypeComparator {
	public:
		/*!
		This method computes, value by value, the relative error between abs(mA) and abs(mB) with a margin and returns
		the number of occured errors.
		Note : abs() is taken because complex elements
		\param mA Eigen matrix representation
		\param mB Eigen matrix representation
		\param margin the relative error margin expressed in %
		\return the number of errors, i.e. the number of elements in mB which have a relative error upper than margin
		*/
		int compare(const ArrayMatrixXcd& mA, const ArrayMatrixXcd& mB, double margin) const;

		/*!
		This method computes, value by value, the relative error between abs(mA) and abs(mB) with a margin and returns
		the number of occured errors.
		Note : abs() is taken because complex elements
		\param mA Eigen matrix representation
		\param mB Eigen matrix representation
		\param margin the relative error margin expressed in %
		\return the number of errors, i.e. the number of elements in mB which have a relative error upper than margin
		*/
		int compare(const VectorMatrixXcd& mA, const VectorMatrixXcd& mB, double margin) const;

		/*!
		This method computes, value by value, the relative error between mA and mB with a margin and returns
		the number of occured errors.
		\param mA Eigen matrix representation
		\param mB Eigen matrix representation
		\param margin the relative error margin expressed in %
		\return the number of errors, i.e. the number of elements in mB which have a relative error upper than margin
		*/
		int compare(const ArrayXXd& mA, const ArrayXXd& mB, double margin) const;

		/*!
		This method computes, value by value, the relative error between mA and mB with a margin and returns
		the number of occured errors.
		\param mA Eigen matrix representation
		\param mB Eigen matrix representation
		\param margin the relative error margin expressed in %
		\return the number of errors, i.e. the number of elements in mB which have a relative error upper than margin
		*/
		int compare(const VectorXd& mA, const VectorXd& mB, double margin) const;

		/*!
		This method computes, value by value, the relative error between mA and mB with a margin and returns
		the number of occured errors.
		\param mA Eigen matrix representation
		\param mB Eigen matrix representation
		\param margin the relative error margin expressed in %
		\return the number of errors, i.e. the number of elements in mB which have a relative error upper than margin
		*/
		int compare(const MatrixXd& mA, const MatrixXd& mB, double margin) const;
	private:
		/*!
		This method computes the relative error between a and b and compares the result to the margin.
		\param a value considered to the true value
		\param b value to be compared
		\param margin the relative error margin expressed in %
		\return 1 if the relative error is upper the margin, 0 otherwise
		*/
		int compareVal(double a, double b, double margin) const {
			return static_cast<int>((abs(a - b) > (margin / 100.)*abs(a)));
		}
	};
}
#endif
