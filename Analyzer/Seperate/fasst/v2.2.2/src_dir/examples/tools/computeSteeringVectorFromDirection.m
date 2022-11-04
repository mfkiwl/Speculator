function A = computeSteeringVectorFromDirection(directions,mic_pos,freq_band_centers)

% This function computes the steering vector of a source at each frequency from its
% direction(s). This gives a matrix A of size I x Arank x F with :
% I : the number of microphones
% Arank : the number of spatial directions
% F : the number of frequency bins
% All equation references in this code refer  : 
% N. Q. K. Duong, E. Vincent and R. Gribonval, "Under-Determined Reverberant
% Audio Source Separation Using a Full-Rank Spatial Covariance Model," 
% in IEEE Transactions on Audio, Speech, and Language Processing, vol. 18, 
% no. 7, pp. 1830-1840, Sept. 2010.
%
% Params :
% directions : Arank x 2 , estimated or true source directions / echoes
% expressed as {azimuth, elevation} in the microphone array referential
% mic_pos : I x 3, microphone locations, in cartesian coordinates
% freq_band_centers : 1 x F, frequency band
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2018 Ewen Camberlein (INRIA), Romain Lebarbenchon (INRIA)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

temperatureC = 24.0;
speedOfSound = 331.4*sqrt(1.0+(temperatureC/273));

I = size(mic_pos,1);
F = length(freq_band_centers);
A_rank = size(directions,1);

% Compute the time tau (eq [9]) between the direction(s) projected on a unit range sphere and each mic of the probe
radius = 1; % 1-meter radius sphere
tau = zeros(I,A_rank);
for rankId=1:A_rank
    tau(:,rankId) = tau_spherical(mic_pos, radius, directions(rankId,1), directions(rankId,2), speedOfSound);
end

% Compute gain kappa, eq [9] 
kappa = 1./( sqrt(4*pi).*tau.*speedOfSound);

% Compute Steering vectors, eq [11]
A = zeros(I,A_rank,F);
for i=1:I
    for rankId=1:A_rank
        A(i,rankId,:) = kappa(i,rankId).*exp(-2*pi*1i*freq_band_centers * tau(i,rankId));
    end
end


end

%% Local functions
function tau=tau_spherical(mic_pos, radius, azimuth, elevation, c)

% This function computes the delay vector between a source and a microphone array, for a given geometry
%
% mic_pos: microphones locations, in cartesien coordinates, in meters
% radius: radius of source location, in spherical coordinates, in meters
% azimuth: azimuth of source location, in spherical coordinates, in degrees
% elevation: elevation of source location, in spherical coordinates, in degrees
% c: sound speed, in m/s
%
% tau: the delays, in seconds, relative to each microphone
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    azimuth = azimuth / 180 * pi;
    elevation = elevation / 180 * pi;
    
    %x = radius .* cos(elevation) .* cos(azimuth);
    %y = radius .* cos(elevation) .* sin(azimuth);
    %z = radius .* sin(elevation);
    [x, y, z] = sph2cart(azimuth, elevation, radius);
    s = repmat([x, y, z], size(mic_pos, 1), 1);
    mean_pos = repmat(mean(mic_pos,1), size(mic_pos, 1), 1);
   
    tau = sqrt(sum((mic_pos - mean_pos - s).^2, 2)); % euclidian distances between the sources and each mic
    tau = tau / c; % compute delays
end