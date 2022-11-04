// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "Source.h"
#include "ERBRepr.h"
#include <sstream>
#include <stdexcept>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <iostream>

using namespace Eigen;
using namespace std;
using namespace tinyxml2;

namespace fasst {
	Source::Source(const XMLElement* xmlNode,int index)
		: m_A(xmlNode->FirstChildElement("A")), m_ex(xmlNode, "ex"), m_ft(xmlNode, "ft") {

		if (xmlNode->FirstChildElement("Wft") == NULL) {
			m_excitationOnly = true;
		}
		else {
			m_excitationOnly = false;
		}

		if (xmlNode->FindAttribute("name") != NULL) {
			m_name = xmlNode->Attribute("name");
		}
		else {
			stringstream ss;
			ss << index;
			m_name = "y" + ss.str();
		}

		double wiener_b;
		if (xmlNode->FirstChildElement("wiener") == NULL) {
			m_wiener_qa = 1;
			wiener_b = 0.;
			m_wiener_c1 = 0;
			m_wiener_c2 = 0;
			m_wiener_qd = 0.;
		}
		else {
			if (xmlNode->FirstChildElement("wiener")->FirstChildElement("a") == NULL) {
				m_wiener_qa = 1;
			}
			else {
				xmlNode->FirstChildElement("wiener")->FirstChildElement("a")->QueryDoubleText(&m_wiener_qa);
				m_wiener_qa = pow(10., m_wiener_qa / 10.); // Convert from dB ; qa=1 if a=0 dB (default value with no effect)
			}

			if (xmlNode->FirstChildElement("wiener")->FirstChildElement("b") == NULL) {
				wiener_b = 0.;
			}
			else {
				xmlNode->FirstChildElement("wiener")->FirstChildElement("b")->QueryDoubleText(&wiener_b);
				if (wiener_b < 0 || wiener_b>1) {
					stringstream s;
					s << "wiener.b = " << wiener_b << " but should be inside [0;1].";
					throw runtime_error(s.str());
				}
			}

			if (xmlNode->FirstChildElement("wiener")->FirstChildElement("c1") == NULL) {
				m_wiener_c1 = 0;
			}
			else {
				xmlNode->FirstChildElement("wiener")->FirstChildElement("c1")->QueryIntText(&m_wiener_c1);
				double c1_test;
				xmlNode->FirstChildElement("wiener")->FirstChildElement("c1")->QueryDoubleText(&c1_test);
				if (m_wiener_c1 < 0) {
					stringstream s;
					s << "wiener.c1 = " << m_wiener_c1 << " but should be >=0.";
					throw runtime_error(s.str());
				}
				if (c1_test != floor(c1_test)) {
					stringstream s;
					s << "wiener.c1 = " << c1_test << " but should be integer";
					throw runtime_error(s.str());
				}
			}
			if (xmlNode->FirstChildElement("wiener")->FirstChildElement("c2") == NULL) {
				m_wiener_c2 = 0;
			}
			else {
				xmlNode->FirstChildElement("wiener")->FirstChildElement("c2")->QueryIntText(&m_wiener_c2);
				double c2_test;
				xmlNode->FirstChildElement("wiener")->FirstChildElement("c2")->QueryDoubleText(&c2_test);
				if (m_wiener_c2 < 0) {
					stringstream s;
					s << "wiener.c2 = " << m_wiener_c2 << " but should be >=0.";
					throw runtime_error(s.str());
				}
				if (c2_test != floor(c2_test)) {
					stringstream s;
					s << "wiener.c2 = " << c2_test << " but should be integer";
					throw runtime_error(s.str());
				}
			}
			if (xmlNode->FirstChildElement("wiener")->FirstChildElement("d") == NULL) {
				m_wiener_qd = 0.;
			}
			else {
				stringstream s;
				s << xmlNode->FirstChildElement("wiener")->FirstChildElement("d")->GetText();
				if (s.str() == "-Inf" || s.str() == "-inf") {
					m_wiener_qd = 0.;
				}
				else {
					xmlNode->FirstChildElement("wiener")->FirstChildElement("d")->QueryDoubleText(&m_wiener_qd);
					m_wiener_qd = pow(10., m_wiener_qd / 10.); // Convert from dB ; qd=0 if d=-Inf dB (default value with no effect)
				}
			}
		}
		if (xmlNode->FirstChildElement("Wex") != NULL) {
			if (xmlNode->FirstChildElement("Wex")->FirstChildElement("rows") != NULL) {
				xmlNode->FirstChildElement("Wex")->FirstChildElement("rows")->QueryIntText(&m_bins);

                if ( (m_bins != m_A.rows()) && (m_A.isConv()) ) {
                    stringstream s;
                    s << "Number of frequency components mismatch : Wex has " << m_bins << " components and A has " << m_A.rows() << " components." <<endl;
                    throw runtime_error(s.str());
                }

			}
			else {
				m_bins = 0;
			}
		}
		else {
			m_bins = 0;
		}

		// Init m_wiener_b matrix
		{
			int I = static_cast<int>(m_A(0).rows());
			m_wiener_b = MatrixXcd::Zero(I, I);
			for (int i1 = 0; i1 < I; i1++) {
				for (int i2 = 0; i2 < I; i2++) {
					if (i1 == i2) {
						m_wiener_b(i1, i2) = 1;
					}
					else {
						m_wiener_b(i1, i2) = wiener_b;
					}
				}
			}
		}

        // compute m_V
		compV();
	}

	void Source::updateSpectralPower(const ArrayXXd &Xi) {
		if (m_excitationOnly) {
			m_ex.update(Xi);
		}
		else {
			ArrayXXd E;
			E = m_ft.V();
			m_ex.update(Xi, E);

			E = m_ex.V();
			m_ft.update(Xi, E);
		}

		compV();
	}

	void Source::compV() {
		if (m_excitationOnly) {
			m_V = m_ex.V();
		}
		else {
			m_V = m_ex.V() * m_ft.V();
		}
	}

    MatrixXcd Source::compSigmaY(int bin, int frame) const {
        if (isInst()) {           
            return m_wiener_qa * m_V(bin, frame) * m_wiener_b * m_A(0) * m_A(0).adjoint();            
        }
        else {            
            return m_wiener_qa * m_V(bin, frame) * m_wiener_b * m_A(bin) * m_A(bin).adjoint();            
        }
    }

	void Source::spectralPowerSmoothing() {
		ArrayXXcd xx = ArrayXXcd::Zero(m_V.rows(), m_V.cols());
		xx.real() = m_V;

		if (m_wiener_c1 != 0) {
			int c1 = m_wiener_c1 * 2 + 1;
			ArrayXcd h1 = ArrayXcd::Zero(c1);
			for (int i1 = 0; i1 < c1; i1++) {
				h1(i1) = 1 + cos(M_PI / (m_wiener_c1 + 1)*(i1 - m_wiener_c1));
			}
			h1 = h1 / h1.sum();
			xx = ERBRepr::fftfilt(h1, xx);
		}

		if (m_wiener_c2 != 0) {
			int c2 = m_wiener_c2 * 2 + 1;
			ArrayXcd h2 = ArrayXcd::Zero(c2);
			for (int i2 = 0; i2 < c2; i2++) {
				h2(i2) = 1 + cos(M_PI / (m_wiener_c2 + 1)*(i2 - m_wiener_c2));
			}
			h2 = h2 / h2.sum();
			xx.transposeInPlace();
			xx = ERBRepr::fftfilt(h2, xx);
			xx.transposeInPlace();
		}
		m_V = xx.real();
	}


	bool Source::equal(const Source & s, double margin) const {
		cout << "Compare A" << endl;
		bool aOk = (*this).A().equal(s.A(), margin);
		cout << "Compare Exitation part (ex)" << endl;
		bool exOk = (*this).getSPEx().equal(s.getSPEx(), margin);
		cout << "Compare Filter part (ft)" << endl;
		bool ftOk = (*this).getSPFt().equal(s.getSPFt(), margin);

		return aOk && exOk && ftOk;
	}

	MatrixXcd Source::WienerThresholding(const MatrixXcd &m, const double& wiener_qd) {
		// Optional thresholding if wiener_qd > 0
		if (wiener_qd > 0) {
			int I = static_cast<int>(m.rows());
			ComplexEigenSolver<MatrixXcd> ces(m);
			MatrixXcd D = ces.eigenvalues().asDiagonal();
			MatrixXcd V = ces.eigenvectors();
			for (int i = 0; i < I; i++) {
				if (abs(D(i, i)) < wiener_qd) {
					D(i, i) = wiener_qd;
				}
			}
			return V * D * V.inverse();
		}
		else {
			return m;
		}

	}

}
