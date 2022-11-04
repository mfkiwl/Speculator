// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "NonNegMatrix.h"
#include <vector>
#include <sstream>

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

namespace fasst {
    NonNegMatrix::NonNegMatrix(const tinyxml2::XMLElement* xmlEl) {
    
        if (xmlEl == NULL) {
        m_adaptability = "";
        _set(MatrixXd(0, 0));
        m_eye = false;
    }
    else {
        int rows;
        int cols;
        // Read attributes
        m_adaptability = xmlEl->Attribute("adaptability", 0);

        // Read dimensions
        xmlEl->FirstChildElement("rows")->QueryIntText(&rows);
        xmlEl->FirstChildElement("cols")->QueryIntText(&cols);

        // Read data
        char *strData = (char *)xmlEl->FirstChildElement("data")->GetText();
        if (strcmp(strData,"eye") == 0) {
            m_eye = true;
        }
        else {
            _set(MatrixXd(rows, cols));
            
            // Parse stream with space delimiter and convert result words to double
            istringstream iss(strData);
            string word;
            for (int i = 0; i < cols; i++) { // col
                for (int j = 0; j < rows; j++) { // rows
                    iss >> word;
                    (*this)(j, i) = atof(word.c_str());
                }
            }
            m_eye = false;
        }
    }
}

    void W::update(const ArrayXXd &Xi, const NonNegMatrix &D) {
  NonNegMatrix Dtranspose = D.transpose();
  ArrayXXd CD = (*this) * D;
  ArrayXXd num = NonNegMatrix(Xi / (CD * CD)) * Dtranspose;
  ArrayXXd denom = NonNegMatrix(1 / CD) * Dtranspose;
  _set(this->array() * (num / denom));
}

	void W::update(const ArrayXXd &Xi, const NonNegMatrix &D, const ArrayXXd &E) {
  NonNegMatrix Dtranspose = D.transpose();
  ArrayXXd CDE = ((*this) * D).array() * E;
  ArrayXXd num = NonNegMatrix(Xi * E / (CDE * CDE)) * Dtranspose;
  ArrayXXd denom = NonNegMatrix(E / CDE) * Dtranspose;
  _set(this->array() * (num / denom));
}

    void UG::update(const ArrayXXd &Xi, const NonNegMatrix &B, const NonNegMatrix &D) {
  NonNegMatrix Btranspose = B.transpose();
  NonNegMatrix Dtranspose = D.transpose();
  ArrayXXd BCD = B * (*this) * D;
  ArrayXXd num = Btranspose * NonNegMatrix(Xi / (BCD * BCD)) * Dtranspose;
  ArrayXXd denom = Btranspose * NonNegMatrix(1 / BCD) * Dtranspose;
  _set(this->array() * (num / denom));
}

    void UG::update(const ArrayXXd &Xi, const NonNegMatrix &B, const NonNegMatrix &D,
                    const ArrayXXd &E) {
  NonNegMatrix Btranspose = B.transpose();
  NonNegMatrix Dtranspose = D.transpose();
  ArrayXXd BCDE = (B * (*this) * D).array() * E;
  ArrayXXd num = Btranspose * NonNegMatrix(Xi * E / (BCDE * BCDE)) * Dtranspose;
  ArrayXXd denom = Btranspose * NonNegMatrix(E / BCDE) * Dtranspose;
  _set(this->array() * (num / denom));
}

    void H::update(const ArrayXXd &Xi, const NonNegMatrix &B) {
  NonNegMatrix Btranspose = B.transpose();
  ArrayXXd BC = B * (*this);
  ArrayXXd num = Btranspose * NonNegMatrix(Xi / (BC * BC));
  ArrayXXd denom = Btranspose * NonNegMatrix(1 / BC);
  _set(this->array() * (num / denom));
}

    void H::update(const ArrayXXd &Xi, const NonNegMatrix &B, const ArrayXXd &E) {
  NonNegMatrix Btranspose = B.transpose();
  ArrayXXd BCE = (B * (*this)).array() * E;
  ArrayXXd num = Btranspose * NonNegMatrix(Xi * E / (BCE * BCE));
  ArrayXXd denom = Btranspose * NonNegMatrix(E / BCE);
  _set(this->array() * (num / denom));
}

    bool NonNegMatrix::equal(const NonNegMatrix & m, double margin) const {
        
        if ((*this).isEye() && m.isEye()) {
            cout << "Content match" << endl;
            return true;
        }
        else {
            bool isOk;
            compare((*this), m, margin) == 0 ? isOk = true : isOk = false;
            return isOk;
        }
    }
}
