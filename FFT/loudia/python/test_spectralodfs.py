#!/usr/bin/env python

import loudia
from common import *
import pylab
import os, sys, wave
import scipy

filename = sys.argv[1]

# Accepted difference between the groundtruth
# and the estimated onsets in milliseconds (ms)
onsetError = 50.0

frameSize = 1024 
frameStep = 512

fftSize = 2048*2
plotSize = fftSize / 8

stream, sampleRate, nframes, nchannels, loader = get_framer_audio(filename, frameSize, frameStep)


# Take all the SpectralODF availables
odfNames = [(getattr(loudia.SpectralODF, i), i) for i in dir(loudia.SpectralODF)
            if type(getattr(loudia.SpectralODF, i)) == int]

odfNames.sort()

specs = []
ffts = []
odfs = {}

ffter = loudia.FFT( fftSize )
windower = loudia.Window( frameSize, loudia.Window.HAMMING )

for frame in stream:
    fft = ffter.process( windower.process( frame ) )[0, :]
    spec =  loudia.magToDb( abs( fft ) )[0, :plotSize]

    ffts.append( fft )
    specs.append( spec )

ffts = scipy.array( ffts )
specs = scipy.array( specs )

frameCount = specs.shape[0] - 1
subplots = len( odfNames ) + 1

# Get the onsets
annotation = os.path.splitext(filename)[0] + '.onset_annotated'
onsets = []
if os.path.isfile(annotation):
    onsets = get_onsets(annotation, frameStep, sampleRate, onsetError = onsetError)
        
pylab.figure()
pylab.hold(True)
pylab.subplot(subplots, 1, 1)

pylab.imshow( scipy.flipud(specs.T), aspect = 'auto' )

draw_onsets( onsets )

pylab.title( 'Spectrogram' )
ax = pylab.gca()

ax.set_xticks( ax.get_xticks()[1:] )
ticks = ax.get_xticks()
ax.set_xticklabels(['%.2f' % (float(tick) * frameStep / sampleRate) for tick in ticks])

ax.set_yticks([])
ax.set_yticklabels([])

ax.set_xlim([0, frameCount - 1])

    
# Create the SpectralODF processors and process
for i, (odfType, odfName) in enumerate(odfNames):
    odf = loudia.SpectralODF( fftSize, odfType )

    print 'Processing: %s...' % odfName
    odfValues = odf.process( ffts )
    print 'Finished'

    pylab.subplot(subplots, 1, i+2)
    pylab.plot(odfValues[:,0])

    draw_onsets( onsets )

    pylab.title( odfName.replace('_', ' ').capitalize() )
    ax = pylab.gca()

    ax.set_xlim([0, frameCount - 1])
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])

pylab.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95, hspace=0.6)
        
pylab.show()
