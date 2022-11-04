//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2010, Alexander Lehmann <lehmanna@in.tum.de>
//                          Felix Weninger <felix@weninger.de>
//                          Bjoern Schuller <schuller@tum.de>
//
// Institute for Human-Machine Communication
// Technische Universitaet Muenchen (TUM), D-80333 Munich, Germany
//
// openBliSSART is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 2 of the License, or (at your option) any later
// version.
//
// openBliSSART is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// openBliSSART.  If not, see <http://www.gnu.org/licenses/>.
//


#include <blissart/audio/MelFilter.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/generators/generators.h>
#include <cassert>


using namespace blissart::linalg;


namespace blissart {


namespace audio {


MelFilter::MelFilter() : 
    _nBands(26), _sampleRate(44100), _lowFreq(0.0), _highFreq(0.0),
    _scaleFactor(1.0),
    _nBins(0), _filterCoeffs(0), _filterIndex(0),
    _lowestIndex(0), _highestIndex(0)
{
}


MelFilter::MelFilter(unsigned int nBands, unsigned int sampleRate,
                     double lowFreq, double highFreq) :
    _nBands(nBands), _sampleRate(sampleRate),
    _lowFreq(lowFreq), _highFreq(highFreq),
    _scaleFactor(1.0),
    _nBins(0), _filterCoeffs(0), _filterIndex(0),
    _lowestIndex(0), _highestIndex(0)
{
}


Matrix* MelFilter::melSpectrum(const Matrix& spectrogram)
{
    computeFilters(spectrogram.rows());

    // Process matrix column by column.
    int m = 0;
    Matrix* rv = new Matrix(_nBands, spectrogram.cols(), generators::zero);
    for (unsigned int j = 0; j < spectrogram.cols(); ++j) {
        for (unsigned int i = _lowestIndex; i <= _highestIndex; ++i) {
            m = _filterIndex[i];
            if (m > -2) {
                double out = spectrogram.at(i, j) * _filterCoeffs[i];
                if (m > -1) {
                    rv->at(m, j) += out;
                }
                if (m < (int)_nBands - 1) {
                    rv->at(m + 1, j) += spectrogram.at(i, j) - out;
                }
            }
        }
        if (_scaleFactor != 1.0) {
            for (m = 0; m < (int)_nBands; ++m) {
                rv->at(m, j) *= _scaleFactor;
            }
        }
    }

    return rv;
}


void MelFilter::synth(const Matrix& melSpectrogram, Matrix& spectrogram)
{
    computeFilters(spectrogram.rows());

    // Compute filter sums for normalization
    double* filterCoeffSums = new double[_nBands];
    for (int fi = 0; fi < _nBands; ++fi) {
        filterCoeffSums[fi] = 0.0;
    }
    for (unsigned int i = _lowestIndex; i <= _highestIndex; ++i) {
        int m = _filterIndex[i];
        if (m > -2) {
            if (m > -1) {
                filterCoeffSums[m] += _filterCoeffs[i];
            }
            if (m < (int)_nBands - 1) {
                filterCoeffSums[m + 1] += (1.0 - _filterCoeffs[i]);
            }
        }
    }
    
    // Compute normalization for each frequency
    double* specNorm = new double[spectrogram.rows()];
    for (unsigned int i = _lowestIndex; i <= _highestIndex; ++i) {
        specNorm[i] = 0.0;
        int m = _filterIndex[i];
        if (m > -2) {
            if (m > -1) {
                specNorm[i] += filterCoeffSums[m] * _filterCoeffs[i];
            }
            if (m < (int)_nBands - 1) {
                specNorm[i] += filterCoeffSums[m + 1] * (1.0 - _filterCoeffs[i]);
            }
        }
        if (_scaleFactor != 1.0)
            specNorm[i] *= _scaleFactor;
        if (specNorm[i] > 0.0)
            specNorm[i] = 1.0 / specNorm[i];
    }

    spectrogram.zero();
    for (unsigned int j = 0; j < spectrogram.cols(); ++j) {
        for (unsigned int i = _lowestIndex; i <= _highestIndex; ++i) {
            int m = _filterIndex[i];
            if (m > -2) {
                if (m > -1) {
                    spectrogram.at(i, j) += melSpectrogram.at(m, j) * _filterCoeffs[i];
                }
                if (m < (int)_nBands - 1) {
                    spectrogram.at(i, j) += melSpectrogram.at(m + 1, j) * (1.0 - _filterCoeffs[i]);
                }
            }
        }
        for (unsigned int i = _lowestIndex; i <= _highestIndex; ++i) {
            spectrogram.at(i, j) *= specNorm[i];
        }
    }

    delete[] specNorm;
    delete[] filterCoeffSums;
}


inline double round(double x) 
{
    return (x > 0.0 ? floor(x + 0.5) : ceil(x - 0.5));
}


void MelFilter::computeFilters(unsigned int nBins)
{
    if (nBins == _nBins) 
        return;

    assert((int)nBins >= _nBands);
    assert(_sampleRate > 0.0);
    assert(_lowFreq >= 0.0 && _highFreq >= 0.0);

    if (_highFreq == 0.0) _highFreq = (double) _sampleRate / 2.0;

    // Precompute basic parameters.
    unsigned int nSamples       = (nBins - 1) * 2;
    const double baseFreq       = (double) _sampleRate / nSamples;
    const double lowestMelFreq  = hertzToMel(_lowFreq);
    const double highestMelFreq = hertzToMel(_highFreq);
    _lowestIndex                = (unsigned int) round(_lowFreq / baseFreq);
    _highestIndex               = (unsigned int) round(_highFreq / baseFreq);
    
    // Always ignore zeroth FFT coefficient (DC component).
    if (_lowestIndex < 1) _lowestIndex = 1;
    
    assert(_highestIndex < nBins && _lowestIndex < nBins);

    // Compute Mel center frequencies
    double* centerFrequencies = new double[_nBands + 2];
    const double halfBw = highestMelFreq / ((double)_nBands + 1.0);
    int m = 0;
    for (; m <= (int)_nBands + 1; ++m) {
        // Distance between center frequencies is half the Mel bandwidth of
        // a filter.
        centerFrequencies[m] = lowestMelFreq + (double)m * halfBw;
    }

    // Allocate memory for filter mapping and coefficients.
    if (_filterIndex)
        delete[] _filterIndex;
    _filterIndex = new int[nBins];
    if (_filterCoeffs)
        delete[] _filterCoeffs;
    _filterCoeffs = new double[nBins];

    // Actually this is NOT necessary, but we want to be ABSOLUTELY sure ...
    for (unsigned int b = 0; b < nBins; ++b) {
        _filterCoeffs[b] = 0.0;
        _filterIndex[b] = -3;
    }

    // Compute the index of the filter that is to be applied to every component
    // of the spectrum. Note that the falling slope of filter M is equal to 
    // the rising slope of filter M+1, which is why we calculate falling slopes
    // only - thus the filter index can have a value of -1. A value of -2 
    // (occurring at the boundaries of the filter bank) indicates that the 
    // filter output is zero.
    m = 0;
    for (unsigned int i = 0; i < nBins; ++i) {
        if (i < _lowestIndex || i > _highestIndex) {
            // XXX: -3?
            _filterIndex[i] = -3;
        }
        else {
            double binFreq = hertzToMel((double)i * baseFreq);
            while (m <= (int)_nBands + 1 && centerFrequencies[m] < binFreq) {
                ++m;
            }
            _filterIndex[i] = m - 2;
        }
    }

    // Compute filter coefficients (for falling slopes).
    m = 0;
    for (unsigned int i = _lowestIndex; i <= _highestIndex; ++i) {
        double binFreq = hertzToMel((double)i * baseFreq);
        while (m <= (int)_nBands && binFreq > centerFrequencies[m + 1]) {
            ++m;
        }
        _filterCoeffs[i] = (centerFrequencies[m + 1] - binFreq) /
                           (centerFrequencies[m + 1] - centerFrequencies[m]);
    }

    // Free memory.
    delete[] centerFrequencies;

    // Save number of bins so that we might save recomputation.
    _nBins = nBins;
}


} // namespace audio


} // namespace blissart

