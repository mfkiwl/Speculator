// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "XMLDoc.h"
#include "Sources.h"
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace tinyxml2;

namespace fasst {
XMLDoc::XMLDoc(const char *fname) {
  // Open fname
	XMLError xmlErr = m_doc.LoadFile(fname);
	if (xmlErr != XML_SUCCESS) {
		stringstream s;
		s << "Can not open " << fname << ". ";
		s << "File probably doesn't exist or isn't readable.";
		throw runtime_error(s.str());
	}
}

int XMLDoc::getIterations() const  {
	int rVal;
	const XMLElement* xmlIterEl = m_doc.FirstChildElement("sources")->FirstChildElement("iterations");
	if (xmlIterEl == NULL) {
		//Default : 0 iterations
		rVal = 0;
	// TODO: Mettre un log
	} else {
		xmlIterEl->QueryIntText(&rVal);
	}
	return rVal;
}

std::string XMLDoc::getTFRType() const  {

	const XMLElement* xmlTfrEl = m_doc.FirstChildElement("sources")->FirstChildElement("tfr_type");
	if (xmlTfrEl == NULL) {
		//Default : STFT assignment
		return "STFT";
	} else {
		return xmlTfrEl->GetText();
	}
}

int XMLDoc::getWlen() const {
  
	int rVal;
	const XMLElement* xmlWlengthEl = m_doc.FirstChildElement("sources")->FirstChildElement("wlen");
	if (xmlWlengthEl == NULL) {
		// Default : 0
		rVal = 0;
	} else { 
		xmlWlengthEl->QueryIntText(&rVal);
	}

	return rVal;
}

int XMLDoc::getNbin() const  {
  
	int rVal;
	const XMLElement* xmlNBinEl = m_doc.FirstChildElement("sources")->FirstChildElement("nbin");
	if (xmlNBinEl == NULL)	{
		// Default : 0
		rVal = 0;
	} else {
		xmlNBinEl->QueryIntText(&rVal);
	}

	return rVal;
}

int XMLDoc::getNbinPerERB() const {
	int rVal;
	const XMLElement* xmlNBinEl = m_doc.FirstChildElement("sources")->FirstChildElement("nbinPerERB_ERBLET");
	if (xmlNBinEl == NULL) {
		// Default : 0
		rVal = 0;
	}
	else {
		xmlNBinEl->QueryIntText(&rVal);
	}

	return rVal;
}

Sources XMLDoc::getSources() const {
	return Sources(m_doc.FirstChildElement("sources"));
}

void XMLDoc::update(const NonNegMatrix& nnMatrix, tinyxml2::XMLElement* xmlOldNode){
	// Convert data to string
	stringstream s;
	for (int i = 0; i < nnMatrix.cols(); i++) {
		for (int j = 0; j < nnMatrix.rows(); j++) {
			s << nnMatrix(j, i) << ' ';
		}
		s << '\n';
	}

	// Replace
	xmlOldNode->FirstChildElement("data")->SetText(s.str().c_str());
}

void XMLDoc::update(const SpectralPower& spectralPower, tinyxml2::XMLElement* xmlOldNode, const string& suffix){
	if (spectralPower.getW().isFree()) {
		update(static_cast<NonNegMatrix> (spectralPower.getW()), xmlOldNode->FirstChildElement(("W" + suffix).c_str()));
	}
	if (spectralPower.getU().isFree()) {
		update(static_cast<NonNegMatrix> (spectralPower.getU()), xmlOldNode->FirstChildElement(("U" + suffix).c_str()));
	}
	if (spectralPower.getG().isFree()) {
		update(static_cast<NonNegMatrix> (spectralPower.getG()), xmlOldNode->FirstChildElement(("G" + suffix).c_str()));
	}
	if (spectralPower.getH().isFree()) {
		update(static_cast<NonNegMatrix> (spectralPower.getH()), xmlOldNode->FirstChildElement(("H" + suffix).c_str()));
	}
}

void XMLDoc::update(const MixingParameter& mixingParam, tinyxml2::XMLElement* xmlOldNode){

	int I = static_cast<int>(mixingParam(0).rows());
	int R = static_cast<int>(mixingParam(0).cols());

	// Convert data to string
	stringstream s;
	if (mixingParam.isInst()) {
		for (int r = 0; r < R; r++) {
			for (int i = 0; i < I; i++) {
				s << mixingParam(0)(i, r).real() << ' ';
			}
		}
	}
	else {
		for (int f = 0; f < mixingParam.size(); f++) {
			for (int r = 0; r < R; r++) {
				for (int i = 0; i < I; i++) {
					s << mixingParam(f)(i, r).real() << ' ';
				}
			}
			for (int r = 0; r < R; r++) {
				for (int i = 0; i < I; i++) {
					s << mixingParam(f)(i, r).imag() << ' ';
				}
			}
		}
		// If type was 'real', replace it with 'complex'
		// TODO: Après un EM, la matrice A convolutive devient complexe nécessairement ? => à vérifier et recoder
		string type = xmlOldNode->FirstChildElement("type")->GetText();
		if (type == "real") {
			xmlOldNode->FirstChildElement("type")->SetText("complex");
		}

	} // end else

	// Write string to new node
	xmlOldNode->FirstChildElement("data")->SetText(s.str().c_str());
}

void XMLDoc::update(const Source& source, tinyxml2::XMLElement* xmlOldNode){
	if (source.A().isFree()) {
		update(source.A(), xmlOldNode->FirstChildElement("A"));

	}
	update(source.getSPEx(), xmlOldNode, "ex");
	if (!source.excitationOnly()) {
		update(source.getSPFt(), xmlOldNode, "ft");
	}
}

void XMLDoc::update(const Sources& sources, tinyxml2::XMLElement* xmlOldNode){
		
	// Replace the old source parameters with the new ones
	XMLElement* xmlEl = xmlOldNode->FirstChildElement("source");
	for (int j = 0; j < sources.sources(); j++) {
		update(sources[j], xmlEl);
		xmlEl = xmlEl->NextSiblingElement("source");
	}

}

void XMLDoc::updateXML(const Sources &sources) {
	
	XMLElement* xmlOldNode = m_doc.FirstChildElement("sources");
	update(sources, xmlOldNode);
}

void XMLDoc::write(const char *fname)  {
	XMLError eResult = m_doc.SaveFile(fname,false);
	if (eResult != XML_SUCCESS) {
		stringstream s;
		s << "Can not write " << fname << " on disk. ";
		throw runtime_error(s.str());
	}
}



}

