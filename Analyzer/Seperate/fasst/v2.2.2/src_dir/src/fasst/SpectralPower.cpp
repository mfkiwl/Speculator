// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "SpectralPower.h"
#include <string>
#include <cstring>
using namespace std;
using namespace Eigen;
using namespace tinyxml2;

namespace fasst {
	
	SpectralPower::SpectralPower(const XMLElement* xmlNode, string suffix)
		: m_W(xmlNode->FirstChildElement(("W" + suffix).c_str())),
		m_U(xmlNode->FirstChildElement(("U" + suffix).c_str())),
		m_G(xmlNode->FirstChildElement(("G" + suffix).c_str())),
		m_H(xmlNode->FirstChildElement(("H" + suffix).c_str())) {}

void SpectralPower::update(const ArrayXXd &Xi) {
  if (m_W.isFree()) {
    NonNegMatrix UGH = m_U * m_G * m_H;
    m_W.update(Xi, UGH);
  }

  if (m_U.isFree()) {
    NonNegMatrix GH = m_G * m_H;
    m_U.update(Xi, m_W, GH);
  }

  if (m_G.isFree()) {
    NonNegMatrix WU = m_W * m_U;
    m_G.update(Xi, WU, m_H);
  }

  if (m_H.isFree()) {
    NonNegMatrix WUG = m_W * m_U * m_G;
    m_H.update(Xi, WUG);
  }
}

void SpectralPower::update(const ArrayXXd &Xi, const ArrayXXd &E) {
  if (m_W.isFree()) {
    NonNegMatrix UGH = m_U * m_G * m_H;
    m_W.update(Xi, UGH, E);
  }

  if (m_U.isFree()) {
    NonNegMatrix GH = m_G * m_H;
    m_U.update(Xi, m_W, GH, E);
  }

  if (m_G.isFree()) {
    NonNegMatrix WU = m_W * m_U;
    m_G.update(Xi, WU, m_H, E);
  }

  if (m_H.isFree()) {
    NonNegMatrix WUG = m_W * m_U * m_G;
    m_H.update(Xi, WUG, E);
  }

}

bool SpectralPower::equal(const SpectralPower & m, double margin) const {

    bool uOk = true;
    bool gOk = true;
    bool wOk = true;
    bool hOk = true;

    cout << "Check U" << endl;
    uOk = (*this).getU().equal(m.getU(), margin);

    cout << "Check G" << endl;
    gOk = (*this).getG().equal(m.getG(), margin);

    cout << "Check W" << endl;
    wOk = (*this).getW().equal(m.getW(), margin);

    cout << "Check H" << endl;
    hOk = (*this).getH().equal(m.getH(), margin);

    return uOk && gOk && wOk && hOk;

}
}
