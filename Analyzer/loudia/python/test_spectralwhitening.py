#!/usr/bin/env python

import loudia
from common import *
import pylab
import os, sys, wave
import scipy

interactivePlot = False

filename = sys.argv[1]

frameSize = 1024
frameStep = 512

fftSize = frameSize * 2
plotSize = fftSize / 4

bandwidth = 4 * fftSize/frameSize
analysisLimit = scipy.inf

stream, sampleRate, nframes, nchannels, loader = get_framer_audio(filename, frameSize, frameStep)

windower = loudia.Window( frameSize, loudia.Window.HAMMING )
ffter = loudia.FFT( fftSize )
whitening = loudia.SpectralWhitening( fftSize, 50.0, 6000.0, sampleRate )

specs = []
results = []
if interactivePlot:
    pylab.ion()
    pylab.figure()
    pylab.title('Interactive plot of the FFT vs LPC frequency response')
    pylab.gca().set_ylim([-100, 40])
    pylab.gca().set_autoscale_on(False)

for frame in stream:
    fft = ffter.process( windower.process( frame ) )
    spec =  loudia.magToDb( abs( fft ) )

    result = whitening.process( abs( fft ) )

    if interactivePlot:
        pylab.subplot(211)
        pylab.hold(False)
        pylab.plot( spec[0, :] )
        
        pylab.subplot(212)
        pylab.hold(False)
        pylab.plot(result[0, :], label = 'Noise Suppressed Spectrum')

    
    specs.append( spec[0, :plotSize] )
    results.append( result[0, :plotSize] )

if interactivePlot:
    pylab.ioff()


specs = scipy.array( specs )
results = scipy.array( results )
frameCount = specs.shape[0] - 1
