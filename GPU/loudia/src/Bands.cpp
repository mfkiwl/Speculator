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

#include "Bands.h"

using namespace std;
using namespace Eigen;

Bands::Bands() : 
  _starts(1, 1)
{ 
  _weights.push_back(MatrixXR::Constant(1, 1, 0.0));

  setup();
}


Bands::Bands(MatrixXI starts, std::vector<MatrixXR> weights) {
  LOUDIA_DEBUG("BANDS: Constructor starts: " << starts);

  if ( starts.rows() != (int)weights.size() ) {
    // Throw an exception
  }

  if ( starts.cols() != 1 ) {
    // Throw an exception
  }


  for (int i = 0; i < (int)weights.size(); i++){
    if ( weights[i].cols() != 1 ) {
      // Throw an exception
    }
  }

  _starts = starts;
  _weights = weights;

  setup();
}

Bands::~Bands() {}

void Bands::setup(){
  // Prepare the buffers
  LOUDIA_DEBUG("BANDS: Setting up...");

  reset();
  LOUDIA_DEBUG("BANDS: Finished set up...");
}


void Bands::process(const MatrixXR& spectrum, MatrixXR* bands){

  (*bands).resize(spectrum.rows(), _starts.rows());

  for (int j = 0; j < spectrum.rows(); j++) {
    for (int i = 0; i < _starts.rows(); i++ ) {
      (*bands)(j, i) = spectrum.block(j, _starts(i, 0), 1, _weights[i].rows()).row(0).dot(_weights[i].col(0));
    }
  }
}

void Bands::reset(){
  // Initial values
}

vector<MatrixXR> Bands::weights() const {
  return _weights;
}

void Bands::bandWeights(int band, MatrixXR* bandWeights) const {
  (*bandWeights) =  _weights[ band ];
}

void Bands::starts(MatrixXI* result) const {
  (*result) = _starts;
}

void Bands::setStartsWeights(const MatrixXI& starts, std::vector<MatrixXR> weights, bool callSetup ) {
  _weights = weights;
  _starts = starts;
  
  if ( callSetup ) setup();
}

int Bands::bands() const {
  return _starts.rows();
}
