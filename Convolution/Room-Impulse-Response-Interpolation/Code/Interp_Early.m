% Name: Karan Pareek
% Net ID: kp2218
%
% This function calculates the interpolation of impulse responses using the
% Dynamic Time Warping method. The output contains two matrices with the 
% left and right channel respectively. 
% 
% INPUT: Input IRs (ir1,ir2), Actual angles (theta1,theta2), Interpolated 
% angle (thetaInt), Dimensions of the room (L,W,H)
% OUTPUT: Left interpolated matrix (int_early_L), Right interpolated matrix (int_early_R)
%
% References
% [1] Kearney, G., Masterson, C., Adams, S., Boland, F. (2009). “Approximation
%     of Binaural Room Impulse Responses.” IET Irish Signals and Systems 
%     Conference, pp. 1-6.
% [2] Kearney, G., Masterson, C., Adams, S., Boland, F. (2009). “Dynamic 
%     Time Warping for Acoustic Response Interpolation: Possibilities and Limitations.” 
%     Signal Processing Conference, 2009 17th European, pp. 705-709.
% [3] Masterson, C., Kearney, G., Boland, F. (2009). “Acoustic Impulse Response 
%     Interpolation for Multichannel Systems using Dynamic Time Warping.” Audio 
%     Engineering Society Conference: 35th International Conference: Audio for Games, 
%     Audio Engineering Society, pp. 1-10.
% [4] Meesawat, K., Hammershøi, D. (2002). “An Investigation on the Transition 
%     from Early Reflections to a Reverberation Tail in a BRIR.” Proceedings of 
%     the 2002 International Conference on Auditory Display, Kyoto, Japan, pp. 1-5.

function [int_early_L,int_early_R] = Interp_Early(ir1,ir2,theta1,theta2,thetaInt,L,W,H)

%% Direct and Early Relections

[xL_E,~,~,~] = Interp_Transition_Time(ir1,L,W,H);
[yL_E,~,~,~] = Interp_Transition_Time(ir2,L,W,H);

[~,xR_E,~,~] = Interp_Transition_Time(ir1,L,W,H);
[~,yR_E,~,~] = Interp_Transition_Time(ir2,L,W,H);

%% Dynamic Time Warping

% Here, we will perform Dynamic Time Warping (DTW) for both channels of the
% impulse responses so that the impulses are time aligned. After these
% impulses have been processed, we can simply interpolate them using alpha
% and beta pararmeters that we will define later in this section.

[~,i_xL,i_yL] = dtw(xL_E, yL_E, length(xL_E)); % Left
[~,i_xR,i_yR] = dtw(xR_E, yR_E, length(xR_E)); % Right


% Here, the warped vectors are :
% -> i_xL,i_yL
% -> i_xR,i_yR

% r1 - Impulse 01
% r2 - Impulse 02
% r3 - Interpolated Impulse

% Warped Signals (Left)
r1_L = xL_E(i_xL);
r2_L = yL_E(i_yL);

% Warped Signals (Right)
r1_R = xR_E(i_xR);
r2_R = yR_E(i_yR);


%% Interpolation

% Here, we carry out the interpolation of the warped signals and the warped
% vectors using linear interpolation techniques. We will introduce m(theta)
% that denotes a hypercardiod response for each impulse.

% Hypercadiod Response
%m1 = abs(0.25 + 0.75*cos(theta1));
%m2 = abs(0.25 + 0.75*cos(theta2));
%mInt = abs(0.25 + 0.75*cos(thetaInt));

% Left
%r1_L = r1_L/m1; % Left Impulse 1
%r2_L = r2_L/m2; % Left Impulse 2

% Here, a vector deifning the distance from the right ear to all the
% speakers should be defined. Based on that, the values of A and B will be
% calculated and put in the formula. A simple way to do this is to divide 
% the space into 4 quadrants (for each ear)

dist = 2.5; 
radius = 0.25;
x = dist - radius;
y = dist + radius;

% Left Ear

% Quadrant (2,3)
% z_a = x:1/359:y;
% Quadrant (4,1)
% z_b = flip(z_a);
% z_left = [z_b(91:180),z_a,z_b(1:90)];

% Right Ear

% Quadrant (4,1)
% z_c = x:1/359:y;
% Quadrant (2,3)
% z_d = flip(z_c);
% z_right = [z_c(91:180),z_d,z_c(1:90)];
%}

int_L = (r1_L + r2_L)/2; % Left Interpolated Impulse! ***
%int_L = int_L * mInt;

% Right
%r1_R = r1_R/m1; % Right Impulse 1
%r2_R = r2_R/m2; % Right Impulse 2

int_R = (r1_R + r2_R)/2; % Right Interpolated Impulse! ***
%int_R = int_R * mInt;

% Here, the vectors that were generated by the distance matrix to time
% align the impulses together will be interpolated. We can simply take the
% average of the vectors [thetaInt = (theta1+theta2)/2]

i_Int_L = floor((i_xL + i_yL)/2); % Left Interpolated Warp Vector ***
i_Int_R = floor((i_xR + i_yR)/2); % Right Interpolated Warp Vector ***

%% Unwarping the Impulses

% Here, we need to use 4 parameters to determine the interpolated left and
% right signals:
% 1. int_L      - Right IR (in the warp domain)
% 2. int_R      - Left IR (in the warp domain)
% 3. i_Int_L    - Right warp vector
% 4. i_Int_R    - Left warp vector

dif_L = diff(i_Int_L);
a1 = dif_L ~= 0;
a1x = int_L(a1);
a1x = a1x';
int_early_L = [ir1(1,1)',a1x]';
%int_early_L = int_early_L/(max(abs(int_early_L)));

dif_R = diff(i_Int_R);
a2 = dif_R ~= 0;
a2x = int_R(a2);
a2x = a2x';
int_early_R = [ir2(1,1)',a2x]';
%int_early_R = int_early_R/(max(abs(int_early_R)));

end