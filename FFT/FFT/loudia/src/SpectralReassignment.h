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

#ifndef SPECTRALREASSIGNMENT_H
#define SPECTRALREASSIGNMENT_H

#include "Typedefs.h"
#include "Debug.h"

#include "Window.h"
#include "FFTComplex.h"

class SpectralReassignment{
protected:
  int _frameSize;
  int _fftSize;
  Real _sampleRate;
  Window::WindowType _windowType;
  
  Window _windowAlgo;
  Window _windowIntegAlgo;
  Window _windowDerivAlgo;

  FFTComplex _fftAlgo;

  MatrixXC _window;
  MatrixXC _windowInteg;
  MatrixXC _windowDeriv;

  MatrixXR _fftAbs2;
  MatrixXC _fftInteg;
  MatrixXC _fftDeriv;

  MatrixXR _time;
  MatrixXR _freq;
 
public: 
  SpectralReassignment(int frameSize, int fftSize, Real sampleRate, Window::WindowType windowType = Window::RECTANGULAR);
  ~SpectralReassignment();
  
  void process(const MatrixXR& frames,
               MatrixXC* fft, MatrixXR* reassignTime, MatrixXR* reassignFreq);
  
  void setup();
  void reset();

  int frameSize() const;
  int fftSize() const;

  Window::WindowType windowType() const;
};

#endif  /* SPECTRALREASSIGNMENT_H */
