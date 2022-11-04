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

#include "SpectralNoiseSuppression.h"
#include "Utils.h"

#include <vector>

using namespace std;
using namespace Eigen;

SpectralNoiseSuppression::SpectralNoiseSuppression(int fftSize, Real lowFrequency, Real highFrequency, Real sampleRate)
{
  LOUDIA_DEBUG("SPECTRALNOISESUPPRESSION: Construction fftSize: " << fftSize
        << " sampleRate: " << sampleRate
        << " lowFrequency: " << lowFrequency
        << " highFrequency: " << highFrequency );

  setFftSize( fftSize, false );
  setSampleRate( sampleRate, false );
  setLowFrequency( lowFrequency, false );
  setHighFrequency( highFrequency, false );

  setup();
}

SpectralNoiseSuppression::~SpectralNoiseSuppression(){}

void SpectralNoiseSuppression::setup(){
  LOUDIA_DEBUG("SPECTRALNOISESUPPRESSION: Setting up...");

  _k0 = (int)(( _lowFrequency / _sampleRate ) * _fftSize);
  _k1 = (int)(( _highFrequency / _sampleRate ) * _fftSize);
  
  // Prepare the bands for the moving average
  int _halfSize = (_fftSize / 2) + 1;

  int minHalfBand = (int)(100.0 / _sampleRate * _fftSize / 2.0);

  MatrixXI starts(_halfSize, 1);
  vector<MatrixXR> weights;
  weights.reserve(_halfSize);
  for ( int i = 0; i < _halfSize; i++ ) {
    int halfBandUnder = max( minHalfBand,  (int)(2.0 / 3.0 * i));
    int halfBandOver = max( minHalfBand,  (int)(i / 2.0 * 3.0));

    int begin = max( i - halfBandUnder, 0 );
    int end = min( i + halfBandOver, _halfSize - 1 );

    starts(i, 0) = begin;

    MatrixXR weight = MatrixXR::Constant(1, end - begin, 1.0 / float(end - begin));
    weights.push_back( weight );
  }
  
  _bands.setStartsWeights( starts, weights );
  _bands.setup();

  reset();

  LOUDIA_DEBUG("SPECTRALNOISESUPPRESSION: Finished setup.");
}

void SpectralNoiseSuppression::process(const MatrixXR& spectrum, MatrixXR* noise, MatrixXR* result){
  const int rows = spectrum.rows();
  const int cols = spectrum.cols();

  (*result).resize(rows, cols);
  
  (*result) = spectrum;

  //DEBUG("SPECTRALNOISESUPPRESSION: Calculate wrapped magnitude.");
  // Calculate the wrapped magnitude of the spectrum
  _g = (1.0 / (_k1 - _k0 + 1.0) * spectrum.block(0, _k0, rows, _k1 - _k0).array().pow(1.0/3.0).rowwise().sum()).array().cube();

  //cout << (_g) << endl;
  
  for ( int i = 0; i < cols; i++ ) {
    (*result).col(i) = (((*result).col(i).array() * _g.array().inverse()) + 1.0).log();
  }
  
  //cout << (*result) << endl;
  
  //DEBUG("SPECTRALNOISESUPPRESSION: Estimate spectral noise.");
  // Estimate spectral noise
  _bands.process((*result), noise);
  
  //DEBUG("SPECTRALNOISESUPPRESSION: Suppress spectral noise.");
  // Suppress spectral noise
  (*result) = ((*result) - (*noise)).array().clipUnder();
}

void SpectralNoiseSuppression::reset(){
  // Initial values

  _bands.reset();
}

Real SpectralNoiseSuppression::lowFrequency() const{
  return _lowFrequency;
}
  
void SpectralNoiseSuppression::setLowFrequency( Real frequency, bool callSetup ){
  _lowFrequency = frequency;
  if ( callSetup ) setup();
}

Real SpectralNoiseSuppression::highFrequency() const{
  return _highFrequency;
}
  
void SpectralNoiseSuppression::setHighFrequency( Real frequency, bool callSetup ){
  _highFrequency = frequency;
  if ( callSetup ) setup();
}

Real SpectralNoiseSuppression::sampleRate() const{
  return _sampleRate;
}
  
void SpectralNoiseSuppression::setSampleRate( Real frequency, bool callSetup ){
  _sampleRate = frequency;
  if ( callSetup ) setup();
}

int SpectralNoiseSuppression::fftSize() const{
  return _fftSize;
}

void SpectralNoiseSuppression::setFftSize( int size, bool callSetup ) {
  _fftSize = size;
  if ( callSetup ) setup();
}
