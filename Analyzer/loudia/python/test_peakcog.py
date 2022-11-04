#!/usr/bin/env python

# Create input
import pylab
import scipy
import loudia

frameSize = 1024
frameStep = 64
beginSize = frameSize
signalSize = 2*frameSize
fftSize = frameSize * (2**2)
sampleRate = 44100

fundamental = 0.100 * sampleRate
harmonics = 20

plotInteractive = False

sine = scipy.zeros(signalSize)
for i in range(harmonics):
    sine[beginSize:] += scipy.cos(2 * scipy.pi * (i+1) * fundamental * scipy.arange(signalSize - beginSize) / sampleRate)

sinenoise = sine + (scipy.random.random(signalSize) - 0.5)

# Loudia's solution # --------------------------------- #
bandwidth = 8 * fftSize / frameSize
peakCount = 1024
window = loudia.Window(frameSize, loudia.Window.BLACKMANHARRIS)
fft = loudia.FFT(fftSize)
peaks = loudia.PeakDetectionComplex()
cogs = loudia.PeakCOG(fftSize, bandwidth)
print peaks.peakCount()

peakCogs = []
peakPos = []
if plotInteractive:
    pylab.ion()
    pylab.hold(False)
    
for i in range((signalSize - frameSize)/frameStep):
    a_sine = sinenoise[i*frameStep:(i*frameStep+frameSize)].T
    
    r_sine_windowed = window.process(a_sine)
    r_sine_fft = fft.process(r_sine_windowed)
    r_sine_peakpos, r_sine_peakmag, r_sine_peakphase = peaks.process(r_sine_fft)
    r_sine_peakcogs = cogs.process(r_sine_fft, r_sine_peakpos)

    if plotInteractive:
        pylab.clf()
        pylab.hold(True)
        pylab.plot(abs(r_sine_fft[0,:]))
        pylab.stem(r_sine_peakpos[0,:], r_sine_peakmag[0,:]) #abs(r_sine_fft[0,:])[map(int, r_sine_peakpos[0,:])])
        
        pylab.hold(False)
        
    peakPos.append(r_sine_peakpos[0,:])
    peakCogs.append(r_sine_peakcogs[0,:])
# -------------------------------------------------------- #
if plotInteractive:
    pylab.ioff()
    
peakCogs = scipy.array(peakCogs)
peakPos = scipy.array(peakPos)
#print peakPos[:10,:]

pylab.figure()
pylab.plot(peakCogs)
pylab.show()


