/*                                                         
** Copyright (C) 2008, 2009 Ricard Marxer <email@ricardmarxer.com>
**                                                                  
** This program is free software; you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation; either version 3 of the License, or   
** (at your option) any later version.                                 
**                                                                     
** This program is distributed in the hope that it will be useful,     
** but WITHOUT ANY WARRANTY; without even the implied warranty of      
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the       
** GNU General Public License for more details.                        
**                                                                     
** You should have received a copy of the GNU General Public License   
** along with this program; if not, write to the Free Software         
** Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA.
*/                                                                          

#ifndef SPECTRALODFMKL_H
#define SPECTRALODFMKL_H

#include "Typedefs.h"
#include "Debug.h"

#include "SpectralODFBase.h"

class SpectralODFMKL : public SpectralODFBase {
protected:
  // Internal parameters
  int _fftSize;
  Real _minSpectrum;
  
  // Internal variables
  MatrixXR _spectrumAbs;

public:
  SpectralODFMKL(int fftSize, Real _minSpectrum = 1e-7);

  ~SpectralODFMKL();

  void setup();

  void process(const MatrixXC& fft, MatrixXR* odfValue);

  void reset();

};

#endif  /* SPECTRALODFMKL_H */
