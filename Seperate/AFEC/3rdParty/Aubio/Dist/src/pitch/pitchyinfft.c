/*
  Copyright (C) 2003-2013 Paul Brossier <piem@aubio.org>

  This file is part of aubio.

  aubio is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  aubio is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with aubio.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "aubio_priv.h"
#include "fvec.h"
#include "cvec.h"
#include "mathutils.h"
#include "spectral/fft.h"
#include "pitch/pitchyinfft.h"

/** pitch yinfft structure */
struct _aubio_pitchyinfft_t
{
  fvec_t *win;        /**< temporal weighting window */
  fvec_t *winput;     /**< windowed spectrum */
  fvec_t *sqrmag;     /**< square difference function */
  fvec_t *weight;     /**< spectral weighting window (psychoacoustic model) */
  fvec_t *fftout;     /**< Fourier transform output */
  aubio_fft_t *fft;   /**< fft object to compute square difference function */
  fvec_t *yinfft;     /**< Yin function */
  smpl_t tol;         /**< Yin tolerance */
  uint_t peak_pos;    /**< currently selected peak pos*/
  uint_t samplerate;  /**< samplerate we got initialized with */
};

static const smpl_t freqs[] = {
     0.,    20.,    25.,   31.5,    40.,    50.,    63.,    80.,   100.,   125.,
   160.,   200.,   250.,   315.,   400.,   500.,   630.,   800.,  1000.,  1250.,
  1600.,  2000.,  2500.,  3150.,  4000.,  5000.,  6300.,  8000.,  9000., 10000.,
 12500., 15000., 20000., 25100
};

static const smpl_t weight[] = {
  -75.8,  -70.1,  -60.8,  -52.1,  -44.2,  -37.5,  -31.3,  -25.6,  -20.9,  -16.5,
  -12.6,  -9.60,  -7.00,  -4.70,  -3.00,  -1.80,  -0.80,  -0.20,  -0.00,   0.50,
   1.60,   3.20,   5.40,   7.80,   8.10,   5.30,  -2.40,  -11.1,  -12.8,  -12.2,
  -7.40,  -17.8,  -17.8,  -17.8
};

aubio_pitchyinfft_t *
new_aubio_pitchyinfft (uint_t samplerate, uint_t bufsize)
{
  uint_t i = 0, j = 1;
  smpl_t freq = 0, a0 = 0, a1 = 0, f0 = 0, f1 = 0;
  aubio_pitchyinfft_t *p = AUBIO_NEW (aubio_pitchyinfft_t);
  p->winput = new_fvec (bufsize);
  p->fft = new_aubio_fft (bufsize);
  p->fftout = new_fvec (bufsize);
  p->sqrmag = new_fvec (bufsize);
  p->yinfft = new_fvec (bufsize / 2 + 1);
  p->tol = 0.85;
  p->peak_pos = 0;
  p->win = new_aubio_window ("hanningz", bufsize);
  p->weight = new_fvec (bufsize / 2 + 1);
  p->samplerate = samplerate;
  for (i = 0; i < p->weight->length; i++) {
    freq = (smpl_t) i / (smpl_t) bufsize *(smpl_t) samplerate;
    while (freq > freqs[j]) {
      j += 1;
    }
    a0 = weight[j - 1];
    f0 = freqs[j - 1];
    a1 = weight[j];
    f1 = freqs[j];
    if (f0 == f1) {           // just in case
      p->weight->data[i] = a0;
    } else if (f0 == 0) {     // y = ax+b
      p->weight->data[i] = (a1 - a0) / f1 * freq + a0;
    } else {
      p->weight->data[i] = (a1 - a0) / (f1 - f0) * freq +
          (a0 - (a1 - a0) / (f1 / f0 - 1.));
    }
    while (freq > freqs[j]) {
      j += 1;
    }
    //AUBIO_DBG("%f\n",p->weight->data[i]);
    p->weight->data[i] = DB2LIN (p->weight->data[i]);
    //p->weight->data[i] = SQRT(DB2LIN(p->weight->data[i]));
  }

  // [taktik] disable weighting
  fvec_set_all (p->weight, 1.0);

  return p;
}

void
aubio_pitchyinfft_do (aubio_pitchyinfft_t * p, const fvec_t * input, fvec_t * output)
{
  uint_t tau, l;
  uint_t length = p->fftout->length;
  uint_t halfperiod;
  fvec_t *fftout = p->fftout;
  fvec_t *yin = p->yinfft;
  smpl_t tmp = 0., sum = 0.;

  // window the input
  fvec_weighted_copy(input, p->win, p->winput);
  // get the real / imag parts of its fft
  aubio_fft_do_complex (p->fft, p->winput, fftout);
  // get the squared magnitude spectrum, applying some weight
  p->sqrmag->data[0] = SQR(fftout->data[0]);
  p->sqrmag->data[0] *= p->weight->data[0];
  for (l = 1; l < length / 2; l++) {
    p->sqrmag->data[l] = SQR(fftout->data[l]) + SQR(fftout->data[length - l]);
    p->sqrmag->data[l] *= p->weight->data[l];
    p->sqrmag->data[length - l] = p->sqrmag->data[l];
  }
  p->sqrmag->data[length / 2] = SQR(fftout->data[length / 2]);
  p->sqrmag->data[length / 2] *= p->weight->data[length / 2];
  // get sum of weighted squared mags
  for (l = 0; l < length / 2 + 1; l++) {
    sum += p->sqrmag->data[l];
  }
  sum *= 2.;
  // get the real / imag parts of the fft of the squared magnitude
  aubio_fft_do_complex (p->fft, p->sqrmag, fftout);
  yin->data[0] = 1.;
  for (tau = 1; tau < yin->length; tau++) {
    // compute the square differences
    yin->data[tau] = sum - fftout->data[tau];
    // and the cumulative mean normalized difference function
    tmp += yin->data[tau];
    if (tmp != 0) {
      yin->data[tau] *= tau / tmp;
    } else {
      yin->data[tau] = 1.;
    }
  }

  // [taktik] changed all below to improve candidate search
  // calc min available confidence first
  tmp = fvec_min(yin);
  if (tmp > p->tol) {
    // give up. got no confident candidate at all 
    p->peak_pos = output->data[0] = 0.;
    return;
  }
  // choose lowest confident candidate first to avoid choosing harmonics
  tau = 0;
  for (l = 1; l < yin->length; l++) {
    // is this candidate "roughly" as good as the lowest one?
    if (ABS (yin->data[l] - tmp) < 0.1) {
      tau = l;
      break;
    }
  }
  // find local min to sharpen the results, seeking rightwards
  for (l = tau + 1; l < yin->length; l++) {
    if (yin->data[l] < yin->data[l - 1]) {
      tau = l;
    } else {
      break;
    }
  }
  p->peak_pos = output->data[0] = fvec_quadratic_peak_pos(yin, tau);
}

void
del_aubio_pitchyinfft (aubio_pitchyinfft_t * p)
{
  del_fvec (p->win);
  del_aubio_fft (p->fft);
  del_fvec (p->yinfft);
  del_fvec (p->sqrmag);
  del_fvec (p->fftout);
  del_fvec (p->winput);
  del_fvec (p->weight);
  AUBIO_FREE (p);
}

smpl_t
aubio_pitchyinfft_get_confidence (aubio_pitchyinfft_t * o) {
  // [taktik] strength of the !currently picked! autocorrelation result
  return 1. - o->yinfft->data[o->peak_pos];
}

uint_t
aubio_pitchyinfft_set_tolerance (aubio_pitchyinfft_t * p, smpl_t tol)
{
  p->tol = tol;
  return 0;
}

smpl_t
aubio_pitchyinfft_get_tolerance (aubio_pitchyinfft_t * p)
{
  return p->tol;
}
