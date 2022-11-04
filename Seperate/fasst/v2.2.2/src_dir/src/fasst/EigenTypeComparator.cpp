#include "EigenTypeComparator.h"
#include <iostream>
using namespace std;
namespace fasst {

	int EigenTypeComparator::compare(const ArrayMatrixXcd& mA, const ArrayMatrixXcd& mB, double margin) const {
		int errorCpt = 0;
		int dim1 = static_cast<int> (mA.rows());
		int dim2 = static_cast<int> (mA.cols());
		int subDim1 = static_cast<int> (mA(0, 0).rows());
		int subDim2 = static_cast<int> (mA(0, 0).cols());

		if (dim1 != static_cast<int> (mB.rows()) ||
			dim2 != static_cast<int> (mB.cols()) ||
			subDim1 != static_cast<int> (mB(0, 0).rows()) ||
			subDim2 != static_cast<int> (mB(0, 0).cols())) {
			cout << "Dimension MISMATCH" << endl;
			return -1;
		}

		for (int i = 0; i < dim1; i++) {
			for (int j = 0; j < dim2; j++) {
				for (int k = 0; k < subDim1; k++) {
					for (int l = 0; l < subDim2; l++) {
						double a = abs(mA(i, j)(k, l));
						double b = abs(mB(i, j)(k, l));
						errorCpt += static_cast<int>(compareVal(a, b, margin));
					}
				}
			}
		}
		if (errorCpt != 0) {
			cout << "Content MISMATCH: " << errorCpt << " errors detected" << endl;
		}

		return errorCpt;
	}

	int EigenTypeComparator::compare(const VectorMatrixXcd& mA, const VectorMatrixXcd& mB, double margin) const {
		int errorCpt = 0;
		int dim1 = static_cast<int> (mA.rows());
		int subDim1 = static_cast<int> (mA(0, 0).rows());
		int subDim2 = static_cast<int> (mA(0, 0).cols());

		if (dim1 != static_cast<int> (mB.rows()) ||
			subDim1 != static_cast<int> (mB(0, 0).rows()) ||
			subDim2 != static_cast<int> (mB(0, 0).cols())) {
			cout << "Dimension MISMATCH" << endl;
			return -1;
		}

		for (int i = 0; i < dim1; i++) {
			for (int k = 0; k < subDim1; k++) {
				for (int l = 0; l < subDim2; l++) {
					double a = abs(mA(i)(k, l));
					double b = abs(mB(i)(k, l));
					errorCpt += static_cast<int>(compareVal(a, b, margin));
				}
			}
		}
		if (errorCpt != 0) {
			cout << "Content MISMATCH: " << errorCpt << " errors detected" << endl;
		}

		return errorCpt;
	}

	int EigenTypeComparator::compare(const ArrayXXd& mA, const ArrayXXd& mB, double margin) const {
		int dim1 = static_cast<int> (mA.rows());
		int dim2 = static_cast<int> (mA.cols());

		if (dim1 != static_cast<int> (mB.rows()) ||
			dim2 != static_cast<int> (mB.cols())) {
			cout << "Dimension MISMATCH" << endl;
			return -1;
		}

		int errorCpt = 0;
		for (int i = 0; i < dim1; i++) {
			for (int j = 0; j < dim2; j++) {
				double a = mA(i, j);
				double b = mB(i, j);
				errorCpt += static_cast<int>(compareVal(a, b, margin));
			}
		}

		if (errorCpt != 0) {
			cout << "Content MISMATCH: " << errorCpt << " errors detected" << endl;
		}
		return errorCpt;
	}

	int EigenTypeComparator::compare(const VectorXd& mA, const VectorXd& mB, double margin) const {
		int errorCpt = 0;
		int dim1 = static_cast<int> (mA.rows());
		if (dim1 != static_cast<int> (mB.rows())) {
			cout << "Dimension MISMATCH" << endl;
			return -1;
		}
		for (int i = 0; i < dim1; i++) {
			double a = mA(i);
			double b = mB(i);
			errorCpt += compareVal(a, b, margin);
		}
		if (errorCpt != 0) {
			cout << "Content MISMATCH: " << errorCpt << " errors detected" << endl;
		}
		return errorCpt;
	}

	int EigenTypeComparator::compare(const MatrixXd& mA, const MatrixXd& mB, double margin) const {
		int errorCpt = 0;
		int dim1 = static_cast<int> (mA.rows());
		int dim2 = static_cast<int> (mB.cols());
		if (dim1 != static_cast<int> (mB.rows()) ||
			dim2 != static_cast<int> (mB.cols())) {
			cout << "Dimension MISMATCH" << endl;
			return -1;
		}
		for (int i = 0; i < dim1; i++) {
			for (int j = 0; j < dim2; j++) {
				double a = mA(i, j);
				double b = mB(i, j);
				errorCpt += compareVal(a, b, margin);
			}
		}
		if (errorCpt != 0) {
			cout << "Content MISMATCH: " << errorCpt << " errors detected" << endl;
		}
		return errorCpt;
	}
}
