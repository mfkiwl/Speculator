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

#ifndef PEAKINTERPOLATIONCOMPLEX_H
#define PEAKINTERPOLATIONCOMPLEX_H

#include "Typedefs.h"
#include "Debug.h"

#include "Utils.h"

/**
 * @class PeakInterpolationComplex
 *
 * @brief Algorithm to interpolate peaks in a vector of Complex values.
 *
 * This class represents an object to interpolate peaks in a vector of Complex values.
 * The algorithm interpolates the positions and magnitudes of a set of peaks, given
 * the original frame, peak positions and peak magnidutes.
 *
 * The interpolation consists in fitting a parabola (quadratic interpolation) on the 
 * point of the peak and the two points surrounding it. 
 *
 * Note that the interpolation is performed in the decibel domain, and in order to 
 * avoid innecessary transformations the resulting interpolated peak magnitudes
 * are returned in decibels.
 *
 * @author Ricard Marxer
 *
 * @sa PeakDetection, PeakDetectionComplex, PeakInterpolation, PeakInterpolationComplex, PeakTracking, PeakTrackingComplex
 */
class PeakInterpolationComplex {
protected:
  // Internal parameters
    
  // Internal variables
  MatrixXR _magnitudes;
  MatrixXR _phases;

public:
  /**
     Constructs a peak interpolation object.
  */
  PeakInterpolationComplex();

  /**
     Destroys the algorithm and frees its resources.
  */
  ~PeakInterpolationComplex();

  void setup();
  void reset();

  /**
     Interpolates the peaks on each of the rows of @a frames, @a peakPositions,
     @a peakMagnitudes, @a peakPhases to put the resulting peak interpolated positions, 
     magnitudes and phases in the rows of @a peakPositions, @a peakMagnitudes and 
     @a peakMagnitudes respectively.

     @param frames matrix of Complex values.
     
     @param peakPositions matrix of Real values (but always Integers) for the peak indices.
     The matrix must have the same number of rows as @a frames and the same number of columns
     as @a peakMagnitudes.
     
     @param peakMagnitudes pointer to a matrix of Real values for the peak magnitudes.
     The matrix must have the same number of rows as @a frames and the same number of columns
     as @a peakPositions and @a peakPhases.

     @param peakPhases pointer to a matrix of Real values for the peak phases.
     The matrix must have the same number of rows as @a frames and the same number of columns
     as @a peakMagnitudes and @a peakPositions.

     @param peakPositionsInterpolated pointer to a matrix of Real values for the peak positions.
     The matrix should have the same number of rows and columns as @a peakPositions, 
     @a peakMagnitudes and @a peakPhases. 
     
     @param peakMagnitudesInterpolated pointer to a matrix of Real values for the peak magnitudes.
     The matrix should have the same number of rows and columns as @a peakPositions, 
     @a peakMagnitudes and @a peakPhases. 
     Note that the units of this matrix are decibels.

     @param peakPhasesInterpolated pointer to a matrix of Real values for the peak phases.
     The matrix should have the same number of rows and columns as @a peakPositions, 
     @a peakMagnitudes and @a peakPhases. 
     
     Note that peaks with positions values smaller than 0.0 (usually -1.0) are not considered peaks and will not
     be interpolated or modified.
     
     Note that if the output matrices are not of the required size they will be resized, 
     reallocating a new memory space if necessary.
  */
  void process(const MatrixXC& frames, 
               const MatrixXR& peakPositions, const MatrixXR& peakMagnitudes, const MatrixXR& peakPhases,
               MatrixXR* peakPositionsInterpolated, MatrixXR* peakMagnitudesInterpolated, MatrixXR* peakPhasesInterpolated);


};

#endif  /* PEAKINTERPOLATIONCOMPLEX_H */
