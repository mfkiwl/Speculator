#!/usr/bin/env python

import loudia
from common import *
import pylab
import os, sys, wave
import scipy

interactivePlot = False
plot = True

filename = sys.argv[1]

frameSize = 2048 
frameStep = 1024

fftSize = 4096
plotSize = fftSize / 4

bandwidth = 4 * fftSize/frameSize
analysisLimit = scipy.inf

freqPrec = 0.001
deltaPeriod = 2
numHarmonics = 10

stream, sampleRate, nframes, nchannels, loader = get_framer_audio(filename, frameSize, frameStep)

windower = loudia.Window( frameSize, loudia.Window.HAMMING )
ffter = loudia.FFT( fftSize )
whitening = loudia.SpectralWhitening(fftSize, 50.0, 6000.0, sampleRate)
pitchSaliency = loudia.PitchSaliency(fftSize, 50.0, 2100.0, sampleRate, freqPrec, numHarmonics)

specs = []
wspecs = []
pitches = []
saliencies = []

if interactivePlot:
    pylab.ion()
    pylab.figure()
    pylab.title('Interactive plot of the FFT vs LPC frequency response')
    pylab.gca().set_ylim([-100, 40])
    pylab.gca().set_autoscale_on(False)

for frame in stream:
    fft = ffter.process( windower.process( frame ) )
    spec =  loudia.magToDb( abs( fft ) )

    wspec = whitening.process( spec )
    pitch, saliency = pitchSaliency.process( wspec )
    
    if interactivePlot:
        pylab.subplot( 211 )
        pylab.hold( False )
        pylab.plot( spec[0, :plotSize] )
        
        pylab.subplot( 212 )
        pylab.hold( False )
        pylab.plot( result[0,:], label = 'Noise Suppressed Spectrum' )
        
    specs.append( spec[0, :plotSize] )
    wspecs.append( wspec[0, :plotSize] )
    pitches.append( pitch )
    saliencies.append( saliency )

if interactivePlot:
    pylab.ioff()


specs = scipy.array( specs )
wspecs = scipy.array( wspecs )
pitches = scipy.array( pitches )[:, 0]
saliencies = scipy.array( saliencies )[:, 0]
frameCount = specs.shape[0] - 1

if plot:
    pylab.figure()
    pylab.subplot(211)
    pylab.plot( pitches[:,0] )

    pylab.subplot(212)
    pylab.plot( saliencies[:,0] )
    pylab.show()

print pitches
