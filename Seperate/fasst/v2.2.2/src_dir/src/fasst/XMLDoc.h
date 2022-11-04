// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#ifndef FASST_XMLDOC_H
#define FASST_XMLDOC_H

#include "tinyxml2.h"
#include <iostream>

namespace fasst {
class Sources;
class Source;
class MixingParameter;
class SpectralPower;
class NonNegMatrix;

/*!
 This class represents an entire DOM and is used to read and write XML file
 */
class XMLDoc {
public:
  /*!
   The main constructor of the class reads a file and parses the XML data inside
   it. Please note that if the file is not readable or not well-formed XML, the
   constructor will throw a `runtime_error` exception.
   \param fname the name of the XML file to be read
   */
  XMLDoc(const char *fname);

  /*!
   \return the number of iterations, or 0 if the field doesn't exist
   */
  
  int getIterations() const ;

  /*!
   \return the window length in the DOM
   */
  std::string getTFRType() const ;

  /*!
   \return the window length in the DOM
   */
  int getWlen() const;

  /*!
   \return the number of frequency bins in the DOM
   */
  int getNbin() const ;

  /*!
  \return the number of bin per ERB in the DOM
  */

  int getNbinPerERB() const;

  /*!
   \return a Sources instance containing all sources in the DOM
   */
  Sources getSources() const;

  /*!
   This updates the sources parameters in the current DOM with updated ones.
   \param sources the updated sources
   */
  void updateXML(const Sources &sources);

  /*!
   This au writes the current DOM to a file. Please note that if the file is not
   writable, the method will throw a `runtime_error` exception.
   \param fname the name of the XML file to be written
   */
  void write(const char *fname) ;
  
private:
    /*! 
    \overload void update(const NonNegMatrix& nnMatrix, tinyxml2::XMLElement* xmlOldNode)
	This method updates XML element with the object data
	\param nnMatrix The Non Negative Matrix object
	\param xmlOldNode The old XML element
	*/
	void update(const NonNegMatrix& nnMatrix, tinyxml2::XMLElement* xmlOldNode);

	/*!
    \overload void update(const SpectralPower& spectralPower, tinyxml2::XMLElement* xmlOldNode, const std::string& suffix)
	This method calls the replace method on each NonNegMatrix parameter which
	degree of adaptability is free.
	\param spectralPower Spectral power object
	\param xmlOldNode The old XML node
	\param suffix Should be either 'ex' or 'ft'
	*/
	void update(const SpectralPower& spectralPower, tinyxml2::XMLElement* xmlOldNode, const std::string& suffix);
	
	/*!
    \overload void update(const MixingParameter& mixingParam, tinyxml2::XMLElement* xmlOldNode)
	This method updates a XML element with the object data
	\param mixingParam Mixing param object
	\param xmlOldNode The old XML element
	*/
	void update(const MixingParameter& mixingParam, tinyxml2::XMLElement* xmlOldNode);

	/*!
    \overload void update(const Sources& sources, tinyxml2::XMLElement* xmlOldNode)
	This method updates each source individual mixing parameter with the global
	mixing parameter, and then replace the XML data with the updated sources.
	\param sources Sources object
	\param xmlOldNode The old XML node
	*/
	void update(const Sources& sources, tinyxml2::XMLElement* xmlOldNode);

	/*!
    \overload void update(const Source& source, tinyxml2::XMLElement* xmlOldNode)
	This method calls the replace method on each parameter whose degree of
	adaptability is free.
	\param source object
	\param xmlOldNode The old XML node
	*/
	void update(const Source& source, tinyxml2::XMLElement* xmlOldNode);

	/*!
	This variable contains the whole DOM
	*/
	tinyxml2::XMLDocument m_doc;

};
}

#endif
