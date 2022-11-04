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

#ifndef UTILS_H
#define UTILS_H

#include "Typedefs.h"
#include "Debug.h"

#include <limits>

/**
 * Given a matrix of polynomes (one per column)
 * returns a matrix of roots (a vector of roots per column)
 */
void roots(const MatrixXR& poly, MatrixXC* result);

/**
 * Given a matrix of roots (a vector of roots per column)
 * returns a matrix of polynomes (a polynome per vector of roots)
 */
void poly(const MatrixXC& roots, MatrixXC* result);

/**
 * Given two row matrices
 * returns the convolution of both
 */
void convolve(const MatrixXC& a, const MatrixXC& b, MatrixXC* c);
void convolve(const MatrixXR& a, const MatrixXR& b, MatrixXR* c);


/**
 * Given two row matrices
 * returns the correlation of both
 */
void correlate(const MatrixXC& a, const MatrixXC& b, MatrixXC* c,
               int _minlag = -std::numeric_limits<int>::infinity(),
               int _maxlag = std::numeric_limits<int>::infinity());

void correlate(const MatrixXR& a, const MatrixXR& b, MatrixXR* c,
               int _minlag = -std::numeric_limits<int>::infinity(),
               int _maxlag = std::numeric_limits<int>::infinity());

/**
 * Given a row matrix
 * returns the autocorrelation
 */
void autocorrelate(const MatrixXR& a, MatrixXR* c,
                   int _minlag = 0,
                   int _maxlag = std::numeric_limits<int>::infinity());

void autocorrelate(const MatrixXC& a, MatrixXC* c,
                   int _minlag = 0,
                   int _maxlag = std::numeric_limits<int>::infinity());


/**
 * Reverse in place the order of the columns
 */
void reverseCols(MatrixXC* in);
void reverseCols(MatrixXR* in);


/**
 * Calculate inplace the cumulative sum
 */
void rowCumsum(MatrixXR* in);
void colCumsum(MatrixXR* in);

/**
 * Calculate inplace shift of a matrix
 */
void rowShift(MatrixXR* in, int num);
void colShift(MatrixXR* in, int num);

/**
 * Calculate inplace range matrix
 */
void range(Real start, Real end, int steps, MatrixXC* in);
void range(Real start, Real end, int steps, int rows, MatrixXC* in);
void range(Real start, Real end, int steps, MatrixXR* in);
void range(Real start, Real end, int steps, int rows, MatrixXR* in);
void range(Real start, Real end, int steps, MatrixXI* in);
void range(Real start, Real end, int steps, int rows, MatrixXI* in);

/**
 * Create a matrix of complex numbers given the polar coordinates
 */
void polar(const MatrixXR& mag, const MatrixXR& phase, MatrixXC* complex);

/**
 * Calculate the combinations of N elements in groups of k
 *
 */
int combination(int N, int k);

/**
 * Calculate the aliased cardinal sine defined as:
 *
 *   asinc(M, T, x) = sin(M * pi * x * T) / sin(pi * x * T)
 */
Real asinc(int M, Real omega);

/**
 * Calculate the Fourier transform of a hamming window
 */
void raisedCosTransform(Real position, Real magnitude,
                        int windowSize, int fftSize,
                        Real alpha, Real beta,
                        MatrixXR* spectrum, int* begin, int* end, int bandwidth);

void raisedCosTransform(Real position, Real magnitude,
                        int windowSize, int fftSize,
                        Real alpha, Real beta,
                        MatrixXR* spectrum, int bandwidth);

void hannTransform(Real position, Real magnitude,
                   int windowSize, int fftSize,
                   MatrixXR* spectrum, int bandwidth = 4);

void hannTransform(Real position, Real magnitude,
                   int windowSize, int fftSize,
                   MatrixXR* spectrum, int* begin, int* end, int bandwidth = 4);


void hammingTransform(Real position, Real magnitude,
                      int windowSize, int fftSize,
                      MatrixXR* spectrum, int bandwidth = 4);

void hammingTransform(Real position, Real magnitude,
                      int windowSize, int fftSize,
                      MatrixXR* spectrum, int* begin, int* end, int bandwidth = 4);

void dbToMag(const MatrixXR& db, MatrixXR* mag);

void magToDb(const MatrixXR& mag, MatrixXR* db, Real minMag = 0.0001 );

void unwrap(const MatrixXR& phases, MatrixXR* unwrapped);

void freqz(const MatrixXR& b, const MatrixXR& a, const MatrixXR& w, MatrixXC* resp);
void freqz(const MatrixXR& b, const MatrixXR& w, MatrixXC* resp);

void derivate(const MatrixXR& a, MatrixXR* b);

int nextPowerOf2(Real a, int factor = 0);

Real gaussian(Real x, Real mu, Real fi);

void gaussian(Real x, MatrixXR mu, MatrixXR fi, MatrixXR* result);

void pseudoInverse(const MatrixXR& a, MatrixXR* result, Real epsilon = 1e-6);

#endif  /* UTILS_H */
