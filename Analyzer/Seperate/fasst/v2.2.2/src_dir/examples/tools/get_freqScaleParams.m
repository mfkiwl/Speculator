% Those functions delas with chosen transform and params to compute nbin
% (number of frequency subbands) and freqBandCenters_Hz (center of each
% frequency subband in Hz) in addition to clean unneeded parameters (set
% to -1)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2018 Ewen Camberlein (INRIA), Romain Lebarbenchon (INRIA)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [freqBandCenters_Hz, nbin, nbinPerERB_ERBLET, nbin_ERB] =  get_freqScaleParams(transformType,wlen,nbin_ERB,nbinPerERB_ERBLET,fs)

switch transformType
    case 'STFT'
        nbin = wlen/2 + 1;
        nbinPerERB_ERBLET = -1;
        nbin_ERB = -1;
        freqBandCenters_Hz = linspace(0.001,fs/2,nbin);
    case 'ERB'
        nbin = nbin_ERB;
        nbin_ERB = nbin_ERB;
        nbinPerERB_ERBLET = -1;
        freqBandCenters_Hz = ERB_computeFreqScaleParams(fs, nbin_ERB);      
    case 'ERBLET'
        nbin_ERB = -1;
        nbinPerERB_ERBLET = nbinPerERB_ERBLET;
        [freqBandCenters_Hz, nbin] = ERBLET_computeFreqScaleParams(fs,nbinPerERB_ERBLET);
    otherwise
        nbin = -1;
        nbin_ERB = -1;
        nbinPerERB_ERBLET = -1;
        freqBandCenters_Hz = -1;
        fprintf('[get_freqScaleParams][Error] Unknown transform -> check your parameters.');
end
end


function freqBandCenters_Hz = ERB_computeFreqScaleParams(fs, nbin_ERB)
fmax=.5*fs; fmin=0;
for j=1:100
    emin=9.26*log(.00437*fmin+1);
    emax=9.26*log(.00437*fmax+1);
    fmin=1.5*(emax-emin)/(nbin_ERB-1)/9.26/.00437*exp(emin/9.26);
    fmax=.5*fs-1.5*(emax-emin)/(nbin_ERB-1)/9.26/.00437*exp(emax/9.26);
    if (fmax < 0) | (fmin > .5*fs), error('The number of frequency bins is too small.'); end
end
% Determining frequency and window length scales
emax=9.26*log(.00437*fmax+1);
e=(0:nbin_ERB-1)*(emax-emin)/(nbin_ERB-1)+emin;
freqBandCenters_Hz=(exp(e/9.26)-1)/.00437;
end

function [fc,nbin] = ERBLET_computeFreqScaleParams(fs, nbinPerERB_ERBLET)
fmin = 0;
fmax = fs/2;

%% Convert fmin and fmax into ERB
erblims = freq2erb([fmin,fmax]);

% Determine number of freq. channels
nbin = nbinPerERB_ERBLET*ceil(erblims(2)-erblims(1));

%% Determine center frequencies
%fc = erbspace(fmin,fmax,Nf)';
fc = erb2freq(linspace(erblims(1),erblims(2),nbin))';
% Set the endpoints to be exactly what the user specified, instead of the
% calculated values
fc(1)=fmin;
fc(end)=fmax;

end

function freq = erb2freq(erb)
freq = (1/0.00437)*sign(erb).*(exp(abs(erb)/9.2645)-1);
end

function erb = freq2erb(freq)
% There is a round-off error in the Glasberg & Moore paper, as
% 1000/(24.7*4.37)*log(10) = 21.332 and not 21.4 as they state.
% The error is tiny, but may be confusing.
% On page 37 of the paper, there is Fortran code with yet another set
% of constants:
%     2302.6/(24.673*4.368)*log10(1+freq*0.004368);
erb = 9.2645*sign(freq).*log(1+abs(freq)*0.00437);
end