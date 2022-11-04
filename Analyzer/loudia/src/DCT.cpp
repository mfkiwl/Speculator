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

#include "DCT.h"

using namespace std;
using namespace Eigen;

DCT::DCT(int inputSize, int dctSize, bool scale, DCTType dctType) :
        _scale( scale )
{
    LOUDIA_DEBUG("DCT: Construction inputSize: " << inputSize
                 << ", dctSize: " << dctSize
                 << ", dctType: " << dctType);

    if (inputSize < dctSize) {
        // TODO: Throw an exception since dctSize is the number of coefficients to output and it cannot output more
        return;
    }

    setInputSize( inputSize, false );
    setDctSize( dctSize, false );
    setDctType( dctType, false );

    setup();
}

DCT::~DCT(){}

void DCT::setup(){
    // Prepare the buffers
    LOUDIA_DEBUG("DCT: Setting up...");
    
    _dctMatrix.resize(_inputSize, _inputSize);

    switch(_dctType) {
    case I:
        type1Matrix( &_dctMatrix );
        break;

    case II:
        type2Matrix( &_dctMatrix );
        break;

    case III:
        // Throw ImplementationError not implemented yet
        break;

    case IV:
        // Throw ImplementationError not implemented yet
        break;

    case OCTAVE:
        typeOctaveMatrix( &_dctMatrix );
        break;

    }


    reset();
    LOUDIA_DEBUG("DCT: Finished setup.");
}

void DCT::type1Matrix(MatrixXR* dctMatrix) {
    int size = (*dctMatrix).rows();

    Real norm = 1.0;
    if ( _scale ) norm = sqrt(Real(2.0)/Real(size - 1));

    for(int i=0; i < size; i++){
        (*dctMatrix)(i, size - 1) = norm * 0.5 * pow((Real)-1, (Real)i);
        for(int j=1; j < size-1; j++){
            (*dctMatrix)(i,j) = norm * cos(Real(j * i) * M_PI / Real(size - 1));
        }
    }

    (*dctMatrix).col(0).setConstant(norm * 0.5);

}

void DCT::type2Matrix(MatrixXR* dctMatrix) {
    int size = (*dctMatrix).rows();

    // In MATLAB the first column is scaled differently
    Real norm0 = 1.0;
    Real norm = 1.0;
    if ( _scale ) {
        norm0 = sqrt(Real(1.0)/Real(size));
        norm = sqrt(Real(2.0)/Real(size));
    }

    for(int i=0; i < size; i++){
        for(int j=0; j < size; j++){
            if (j==0) {
                (*dctMatrix)(i,j) = norm0 * cos(Real(j) * M_PI / Real(size) * (Real(i) + 0.5));
            } else {
                (*dctMatrix)(i,j) = norm * cos(Real(j) * M_PI / Real(size) * (Real(i) + 0.5));
            }
        }
    }
}

void DCT::typeOctaveMatrix(MatrixXR* dctMatrix) {
    int size = (*dctMatrix).rows();

    Real norm = 1.0;
    if ( _scale ) norm = sqrt(2.0/Real(size));

    for(int i=0; i < size; i++){
        for(int j=1; j < size; j++){
            (*dctMatrix)(i,j) = norm * cos(Real(j) * M_PI / Real(2 * size) * (Real(2 * i - 1)));
        }
    }

    (*dctMatrix).col(0).setConstant( norm * sqrt(0.5) );
}

void DCT::process(const MatrixXR& input, MatrixXR* dctCoeffs){
    (*dctCoeffs).resize(input.rows(), _dctSize);

    for ( int i = 0 ; i < input.rows(); i++) {
        (*dctCoeffs).row(i) = input.row(i) * _dctMatrix.block(0, 0, input.cols(), _dctSize);
    }
}

void DCT::reset(){
    // Initial values
}

DCT::DCTType DCT::dctType() const{
    return _dctType;
}

void DCT::setDctType( DCTType type, bool callSetup ) {
    if (type == _dctType) {
      return;
    }
  
    _dctType = type;
    
    if ( callSetup ) setup();
}

int DCT::inputSize() const{
    return _inputSize;
}

void DCT::setInputSize( int size, bool callSetup ) {
    if (size == _inputSize) {
      return;
    }
    
    _inputSize = size;
    if ( callSetup ) setup();
}

int DCT::dctSize() const{
    return _dctSize;
}

void DCT::setDctSize( int size, bool /*callSetup*/ ) {
    _dctSize = size;
}
