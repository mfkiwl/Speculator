#!/usr/bin/python
#
# Those functions deals with chosen transform and params to compute nbin
# (number of frequency subbands) and freqBandCenters_Hz (center of each
# frequency subband in Hz) in addition to clean unneeded parameters (set
# to -1)
# 
###########################################################################
# Copyright 2018 Ewen Camberlein (INRIA), Romain Lebarbenchon (INRIA)
# This software is distributed under the terms of the GNU Public License
# version 3 (http://www.gnu.org/licenses/gpl.txt)
###########################################################################
import numpy as np

def run(transformType,wlen,nbin_ERB,nbinPerERB_ERBLET,fs):

    if transformType == 'STFT':
        nbin = wlen/2 + 1
        nbinPerERB_ERBLET = -1
        nbin_ERB = -1
        freqBandCenters_Hz = np.linspace(0,fs/2,num=nbin)
    elif transformType =='ERB':
        nbin = nbin_ERB
        nbin_ERB = nbin_ERB
        nbinPerERB_ERBLET = -1
        freqBandCenters_Hz = ERB_computeFreqScaleParams(fs, nbin_ERB)       
    elif transformType == 'ERBLET':
        nbin_ERB = -1
        nbinPerERB_ERBLET = nbinPerERB_ERBLET
        freqBandCenters_Hz, nbin = ERBLET_computeFreqScaleParams(fs,nbinPerERB_ERBLET)
    else :
        nbin = -1
        nbin_ERB = -1
        nbinPerERB_ERBLET = -1
        freqBandCenters_Hz = -1
        print '[get_freqScaleParams][Error] Unknown transform -> check your parameters.'
    return [freqBandCenters_Hz, nbin, nbinPerERB_ERBLET, nbin_ERB]
    
def ERB_computeFreqScaleParams(fs, nbin_ERB):
    fmin=0
    fmax=.5*fs
    for j in range (1,100):
        emin=9.26*np.log(.00437*fmin+1)
        emax=9.26*np.log(.00437*fmax+1)
        fmin=1.5*(emax-emin)/(nbin_ERB-1)/9.26/.00437*np.exp(emin/9.26);
        fmax=.5*fs-1.5*(emax-emin)/(nbin_ERB-1)/9.26/.00437*np.exp(emax/9.26);
        if (fmax < 0) | (fmin > .5*fs):
            error('The number of frequency bins is too small.')
            
    # Determining frequency and window length scales
    emax=9.26*np.log(.00437*fmax+1);
    e=np.linspace(0,nbin_ERB-1, num =nbin_ERB)*(emax-emin)/(nbin_ERB-1)+emin
    freqBandCenters_Hz=(np.exp(e/9.26)-1)/.00437;
    return freqBandCenters_Hz

def ERBLET_computeFreqScaleParams(fs, nbinPerERB_ERBLET):
    fmin = 0
    fmax = fs/2

    # Convert fmin and fmax into ERB
    erblims = freq2erb([fmin,fmax])

    # Determine number of freq. channels
    nbin = nbinPerERB_ERBLET*int(np.ceil(erblims[1]-erblims[0]))

    # Determine center frequencies
    fc = erb2freq(np.linspace(erblims[0],erblims[1],num=nbin))
    # Set the endpoints to be exactly specified values, instead of the
    # calculated values   
    fc[0]=fmin
    fc[nbin-1]=fmax
    return [fc,nbin]

def erb2freq(erb) :
    freq = np.multiply((1/0.00437)*np.sign(erb), (np.exp(np.abs(erb)/9.2645)-1))
    return freq


def freq2erb(freq):
    # There is a round-off error in the Glasberg & Moore paper, as
    # 1000/(24.7*4.37)*log(10) = 21.332 and not 21.4 as they state.
    # The error is tiny, but may be confusing.
    # On page 37 of the paper, there is Fortran code with yet another set
    # of constants:
    #     2302.6/(24.673*4.368)*log10(1+freq*0.004368)
    erb = np.multiply(9.2645*np.sign(freq), np.log(1+np.abs(freq)*0.00437))
    return erb