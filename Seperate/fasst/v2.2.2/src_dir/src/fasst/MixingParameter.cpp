// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "MixingParameter.h"
#include <stdexcept>
#include <sstream>

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

namespace fasst {
    MixingParameter::MixingParameter(const XMLElement* xmlEl) {
  // Read attributes
        m_adaptability = xmlEl->Attribute("adaptability", 0);
        m_mixingType = xmlEl->Attribute("mixing_type", 0);

    // Read dimensions
    int ndims;
    xmlEl->FirstChildElement("ndims")->QueryIntText(&ndims);
    if (ndims != 2 && ndims != 3) {
        throw runtime_error("Check your mixing parameter: ndims should be 2 or 3");
    }
    vector<int> dim(ndims);
    const XMLElement* dimEl = xmlEl->FirstChildElement("dim");
    for (int i = 0; i < ndims; i++) {
        dimEl->QueryIntText(&dim[i]);
        dimEl = dimEl->NextSiblingElement("dim");
    }
    // Read type
    string type = xmlEl->FirstChildElement("type")->GetText();
    if (type != "real" && type != "complex") {
        throw runtime_error(
            "Check your mixing parameter: type should be real or complex");
    }

    // Check if mixingType, ndims and type are consistent
    if (m_mixingType == "conv" && ndims != 3) {
        throw runtime_error("Check your mixing parameter: if mixing type is conv, "
            "ndims should be equal to 3");
    }
    else if (m_mixingType == "inst" && ndims != 2) {
        throw runtime_error("Check your mixing parameter: if mixing type is inst, "
            "ndims should be equal to 2");
    }
    else if (m_mixingType == "inst" && type != "real") {
        throw runtime_error("Check your mixing parameter: if mixing type is inst, "
            "type should be real");
    }

    // Read data and convert to double
    int nEl = 0; // number of double element in strData
    if (m_mixingType == "inst") {
        //element are only real and ndims = 2
        nEl = dim[0] * dim[1];
    }
    else if (m_mixingType == "conv") {
        // elements should be real or complex / ndims = 3
        // If complex elements, number of element is doubled (real part and imaginary part)
        nEl = dim[0] * dim[1] * dim[2] * (type == "complex" ? 2 : 1);
    }
    
    // Parse stream with space delimiter and convert result words to double
    char *strData = (char*)xmlEl->FirstChildElement("data")->GetText();
    vector<double> data(nEl);
    istringstream iss(strData);
    string word;
    for (int i = 0; i < nEl; i++){
        iss >> word;
        data[i] = atof(word.c_str());
    }

    if (m_mixingType == "inst") {
        _set(VectorMatrixXcd(1));
        (*this)(0) = MatrixXcd::Zero(dim[0], dim[1]);
        (*this)(0).real() = Eigen::Map<MatrixXd>(&data[0], dim[0], dim[1]);
    }
    else if (m_mixingType == "conv" && type == "real") {
        _set(VectorMatrixXcd(dim[2]));
        int s = dim[0] * dim[1];
        for (int i = 0; i < size(); i++) {
            (*this)(i) = MatrixXcd::Zero(dim[0], dim[1]);
            (*this)(i).real() = Eigen::Map<MatrixXd>(&data[i * s], dim[0], dim[1]);
        }
    }
    else if (m_mixingType == "conv" && type == "complex") {
        _set(VectorMatrixXcd(dim[2]));
        int s = dim[0] * dim[1];
        for (int i = 0; i < size(); i++) {
            (*this)(i) = MatrixXcd(dim[0], dim[1]);

            int index = i * 2 * s;
            (*this)(i).real() = Eigen::Map<MatrixXd>(&data[index], dim[0], dim[1]);

            index = (i * 2 + 1) * s;
            (*this)(i).imag() = Eigen::Map<MatrixXd>(&data[index], dim[0], dim[1]);
        }
    }
}
    
    complex<double> MixingParameter::getVal(int freqId, int channelId, int rankId) const {
        if (isInst()){
            // freqId is ignored
            return (*this)(0)(channelId, rankId);
        } else {
            return (*this)(freqId)(channelId, rankId);
        }
    }

    bool MixingParameter::equal(const MixingParameter & m, double margin) const {
        bool typeOk = ((*this).isConv() && m.isConv()) || ((*this).isInst() && m.isInst());
        if (!typeOk) {
            cout << "Type MISMATCH" << endl;
            return false;
        }
        else {
            bool checkOk;
            (compare((*this), m, margin) == 0 ) ? checkOk = true : checkOk = false;
            return checkOk;
        }
    }
}
