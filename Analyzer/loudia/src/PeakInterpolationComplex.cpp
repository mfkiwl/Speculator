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

#include "PeakInterpolationComplex.h"

using namespace std;
using namespace Eigen;

PeakInterpolationComplex::PeakInterpolationComplex() {
  LOUDIA_DEBUG("PEAKINTERPOLATIONCOMPLEX: Constructor");
  
  setup();

  LOUDIA_DEBUG("PEAKINTERPOLATIONCOMPLEX: Constructed");
}

PeakInterpolationComplex::~PeakInterpolationComplex() {
  // TODO: Here we should free the buffers
  // but I don't know how to do that with MatrixXR and MatrixXR
  // I'm sure Nico will...
}


void PeakInterpolationComplex::setup(){
  // Prepare the buffers
  LOUDIA_DEBUG("PEAKINTERPOLATIONCOMPLEX: Setting up...");

  reset();

  LOUDIA_DEBUG("PEAKINTERPOLATIONCOMPLEX: Finished set up...");
}


void PeakInterpolationComplex::process(const MatrixXC& input,
                              const MatrixXR& peakPositions, const MatrixXR& peakMagnitudes, const MatrixXR& peakPhases,
                              MatrixXR* peakPositionsInterp, MatrixXR* peakMagnitudesInterp, MatrixXR* peakPhasesInterp) {
  
  LOUDIA_DEBUG("PEAKINTERPOLATIONCOMPLEX: Processing");  
  Real leftMag, leftPhase;
  Real rightMag, rightPhase;
  Real mag, interpFactor;
  
  (*peakPositionsInterp).resize(input.rows(), peakPositions.cols());
  (*peakMagnitudesInterp).resize(input.rows(), peakPositions.cols());
  (*peakPhasesInterp).resize(input.rows(), peakPositions.cols());
  
  _magnitudes = input.array().abs();
  unwrap(input.array().angle(), &_phases);
  
  for ( int row = 0 ; row < _magnitudes.rows(); row++ ) {
  
    for ( int i = 0; i < peakPositions.cols(); i++ ) {
      
      // If the position is -1 do nothing since it means it is nothing
      if( peakPositions(row, i) == -1 ){

        (*peakMagnitudesInterp)(row, i) = peakMagnitudes(row, i);         
        (*peakPhasesInterp)(row, i) = peakPhases(row, i); 
        (*peakPositionsInterp)(row, i) = peakPositions(row, i);
        
      } else {
        
        // Take the center magnitude in dB
        mag = 20.0 * log10( peakMagnitudes(row, i) );

        // Take the left magnitude in dB
        if( peakPositions(row, i) <= 0 ){
          
          leftMag = 20.0 * log10( _magnitudes(row, (int)peakPositions(row, i) + 1) );
          
        } else {
          
          leftMag = 20.0 * log10( _magnitudes(row, (int)peakPositions(row, i) - 1) );
          
        }
        
        // Take the right magnitude in dB
        if( peakPositions(row, i) >= _magnitudes.row(row).cols() - 1 ){
          
          rightMag = 20.0 * log10( _magnitudes(row, (int)peakPositions(row, i) - 1) );
          
        } else {
          
          rightMag = 20.0 * log10( _magnitudes(row, (int)peakPositions(row, i) + 1) );
          
        }
                
        // Calculate the interpolated position
        (*peakPositionsInterp)(row, i) = peakPositions(row, i) + 0.5 * (leftMag - rightMag) / (leftMag - 2.0 * mag + rightMag);

        interpFactor = ((*peakPositionsInterp)(row, i) - peakPositions(row, i));

        // Calculate the interpolated magnitude in dB
        (*peakMagnitudesInterp)(row, i) = mag - 0.25 * (leftMag - rightMag) * interpFactor;

        // Calculate the interpolated phase
        leftPhase = _phases(row, (int)floor((*peakPositionsInterp)(row, i)));
        rightPhase = _phases(row, (int)floor((*peakPositionsInterp)(row, i)) + 1);
        
        interpFactor = (interpFactor >= 0) ? interpFactor : interpFactor + 1;
        
        (*peakPhasesInterp)(row, i) = (leftPhase + interpFactor * (rightPhase - leftPhase));
      }
    }
  }

  // Calculate the princarg() of the phase: remap to (-pi pi]
  (*peakPhasesInterp) = ((*peakPhasesInterp).array() != -1).select(((*peakPhasesInterp).array() + M_PI).array().modN(-2.0 * M_PI) + M_PI, (*peakPhasesInterp));
  
  LOUDIA_DEBUG("PEAKINTERPOLATIONCOMPLEX: Finished Processing");
}

void PeakInterpolationComplex::reset(){
  // Initial values
}

