#!/usr/bin/env python

# Create input
import scipy
import loudia

plot = False
frameSize = 256
fftSize = 512
sampleRate = 8000

a_zeros = scipy.zeros(frameSize)
a_ones = scipy.ones(frameSize)
a_random = scipy.random.random(frameSize)
a_sine = scipy.cos(2 * scipy.pi * 440 * scipy.arange(frameSize) / sampleRate + scipy.pi/4.0)

# Loudia's solution # --------------------------------- #
w = loudia.Window(frameSize, loudia.Window.HAMMING)
m = loudia.FFT(fftSize, False)
n = loudia.IFFT(fftSize, False)

r_zeros = n.process(m.process(w.process(a_zeros)))[:,:frameSize]
r_ones = n.process(m.process(w.process(a_ones)))[:,:frameSize]
r_random = n.process(m.process(w.process(a_random)))[:,:frameSize]
r_sine = n.process(m.process(w.process(a_sine)))[:,:frameSize]
# -------------------------------------------------------- #

x_zeros = w.process(a_zeros)
x_ones = w.process(a_ones)
x_random = w.process(a_random)
x_sine = w.process(a_sine)

atol = 1e-6

print scipy.allclose(r_zeros, x_zeros, atol = atol)
print scipy.allclose(r_ones, x_ones, atol = atol)
print scipy.allclose(r_random, x_random, atol = atol)
print scipy.allclose(r_sine, x_sine, atol = atol)

if plot:
    import pylab
    pylab.subplot(211)
    pylab.hold(True)
    pylab.plot(r_sine.T, label = 'Loudia')
    pylab.plot(x_sine.T, label = 'Expected')

    pylab.legend()

    
    pylab.subplot(212)
    pylab.hold(True)
    pylab.plot(r_sine.T - x_sine.T, label = 'Difference')
    
    pylab.legend()
    pylab.show()
