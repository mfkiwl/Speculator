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

#include "Typedefs.h"
#include "Debug.h"

#include "Window.h"
#include "SpectralReassignment.h"
#include "Utils.h"

using namespace std;
using namespace Eigen;

SpectralReassignment::SpectralReassignment(int frameSize, int fftSize, Real sampleRate, Window::WindowType windowType) : 
  _frameSize( frameSize ),
  _fftSize( fftSize ),
  _sampleRate( sampleRate ),
  _windowType( windowType ),
  _windowAlgo( frameSize, windowType ), 
  _windowIntegAlgo( frameSize, Window::CUSTOM ), 
  _windowDerivAlgo( frameSize, Window::CUSTOM ), 
  _fftAlgo( frameSize, fftSize, true )
{
  
  LOUDIA_DEBUG("SPECTRALREASSIGNMENT: Constructor frameSize: " << frameSize << \
        ", fftSize: " << fftSize << \
        ", sampleRate: " << sampleRate << \
        ", windowType: " << windowType);


  setup();
  LOUDIA_DEBUG("SPECTRALREASSIGNMENT: Constructed");
}

SpectralReassignment::~SpectralReassignment(){}

void SpectralReassignment::setup(){
  LOUDIA_DEBUG("SPECTRALREASSIGNMENT: Setting up...");
  
  // Setup the window so it gets calculated and can be reused
  _windowAlgo.setup();
  
  // Create the time vector
  LOUDIA_DEBUG("SPECTRALREASSIGNMENT: Creating time vector...");
  Real timestep = 1.0 / _sampleRate;

  // The unit of the vectors is Time Sample fractions
  // So the difference between one coeff and the next is 1
  // and the center of the window must be 0, so even sized windows
  // will have the two center coeffs to -0.5 and 0.5
  // This should be a line going from [-(window_size - 1)/2 ... (window_size - 1)/2]
  _time.resize(_frameSize, 1);
  for(int i = 0; i < _time.rows(); i++){
    _time(i, 0) = (i - Real(_time.rows() - 1)/2.0);
  }
  
  // Create the freq vector
  LOUDIA_DEBUG("SPECTRALREASSIGNMENT: Creating freq vector...");
  
  // The unit of the vectors is Frequency Bin fractions
  // TODO: Must rethink how the frequency vector is initialized
  // as we did for the time vector
  _freq.resize(1, _fftSize);
  range(0, _fftSize, _fftSize, &_freq);
  
  // Calculate and set the time weighted window
  LOUDIA_DEBUG("SPECTRALREASSIGNMENT: Calculate time weighted window...");
  MatrixXR windowInteg = _windowAlgo.window();
  windowInteg = windowInteg.array() * _time.transpose().array();
  _windowIntegAlgo.setWindow(windowInteg);

  // Calculate and set the time derivated window
  LOUDIA_DEBUG("SPECTRALREASSIGNMENT: Calculate time derivative window...");
  MatrixXR windowDeriv = _windowAlgo.window();
  for(int i = windowDeriv.cols() - 1; i > 0; i--){
    windowDeriv(0, i) = (windowDeriv(0, i) - windowDeriv(0, i - 1)) / timestep;
  }

  // TODO: Check what is the initial condition for the window
  // Should this be 0 or just the value it was originally * dt
  //windowDeriv(0, 0) = 0.0;
  _windowDerivAlgo.setWindow(windowDeriv);

  // Create the necessary buffers for the windowing
  _window.resize(1, _frameSize);
  _windowInteg.resize(1, _frameSize);
  _windowDeriv.resize(1, _frameSize);

  // Create the necessary buffers for the FFT
  _fftAbs2.resize(1, _fftSize);
  _fftInteg.resize(1, _fftSize);
  _fftDeriv.resize(1, _fftSize);
  
  // Setup the algos
  _windowIntegAlgo.setup();
  _windowDerivAlgo.setup();
  _fftAlgo.setup();

  LOUDIA_DEBUG("SPECTRALREASSIGNMENT: Finished set up...");
}


void SpectralReassignment::process(const MatrixXR& frames,
                                   MatrixXC* fft, MatrixXR* reassignTime, MatrixXR* reassignFreq){
  
  // Process the windowing
  _windowAlgo.process(frames, &_window);
  _windowIntegAlgo.process(frames, &_windowInteg);
  _windowDerivAlgo.process(frames, &_windowDeriv);
  
  // Process the FFT
  _fftAlgo.process(_window, fft);
  _fftAlgo.process(_windowInteg, &_fftInteg);
  _fftAlgo.process(_windowDeriv, &_fftDeriv);
  
  // Create the reassignment operations
  _fftAbs2 = (*fft).array().abs2();

  // Create the reassign operator matrix
  // TODO: check if the current timestamp is enough for a good reassignment
  // we might need for it to depend on past frames (if the reassignment of time
  // goes further than one)
  LOUDIA_DEBUG("SPECTRALREASSIGNMENT: Processing: creating the time reassignment operation...");    
  (*reassignTime) = -((_fftInteg.array() * (*fft).conjugate().array()) / _fftAbs2.cast<Complex>().array()).real();
    
  // TODO: Check the unity of the freq reassignment, it may need to be normalized by something
  LOUDIA_DEBUG("SPECTRALREASSIGNMENT: Processing: creating the freq reassignment operation...");
  (*reassignFreq) = _freq.array() + ((_fftDeriv.array() * (*fft).conjugate().array()) / _fftAbs2.cast<Complex>().array()).imag();
  
  (*reassignTime) = ((*reassignTime).array().isNaN()).matrix().select(0, (*reassignTime));
  (*reassignFreq) = ((*reassignFreq).array().isNaN()).matrix().select(0, (*reassignFreq));
  
  // Reassign the spectrum values
  // TODO: put this into a function and do it right
  // will have to calculate and return all the reassigned values:
  // reassignedFrequency, reassignedTime:
  //      - are calculated using Flandrin's method using the 3 DFT
  // reassignedMagnitude, reassignedPhase: 
  //      - are calculated from reassigned freq and time and the original DFT
  //        (the magnitude and phase must then be put back 
  //         in the form of a complex in the reassigned frame)
  /*
  LOUDIA_DEBUG("SPECTRALREASSIGNMENT: Processing: reassigning...");
  LOUDIA_DEBUG("SPECTRALREASSIGNMENT: Processing: reassigning _reassignFreq: " << _reassignFreq.rows() << ", " << _reassignFreq.cols());
  
  for(int j = 0; j < _reassignFreq.cols(); j++){
    
    if((int)round(_reassignFreq(i, j)) >= 0 && (int)round(_reassignFreq(i, j)) < (*reassigned).cols()) {
      
      (*reassigned)(i, (int)round(_reassignFreq(i, j))) += ((1.0 - (abs(_reassignFreq(i, j) - (int)round(_reassignFreq(i,j))))) * abs(_fft(i, (int)round(_reassignFreq(i,j)))));
      
    }
  }
  */
}

void SpectralReassignment::reset(){
  _windowAlgo.reset();
  _windowIntegAlgo.reset();
  _windowDerivAlgo.reset();
  _fftAlgo.reset();
}

int SpectralReassignment::frameSize() const{
  return _frameSize;
}

int SpectralReassignment::fftSize() const{
  return _fftSize;
}

Window::WindowType SpectralReassignment::windowType() const{
  return _windowType;
}
