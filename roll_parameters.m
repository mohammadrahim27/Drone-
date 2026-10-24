J = 2572131.00 * 10^-9; % roll inertia (Ixx) taken from CAD model

%% Rate loop (inner PI)

% Gains (tuned by us)
kp_in_rate = 20 * J;
ki_in_rate = 2 * kp_in_rate; 

%% Angle loop (outer P)

% Gains (tuned by us)
zeta = 1; % critically damped
kp_out_rate = ((20 * J) / J) / (4 * (zeta^2));


rate_max = deg2rad(200); % ~ 3.5 rad/s (set from PX4, to verify)
% without rate_max, large angle errors create huge rate commands -- this
% gets clipped to 3.5 rad/s so drone never rotates faster than designed for

%% Test inputs
rate_step = 0.52; % rad/s -- simulates pilot roll rate (ex: roll at 30 deg/s)
% check time constant = 0.05s, no overshoot

angle_step = 0.2; % rad -- simulates pilot tilt angle (ex: tilt 11 deg right)
% to check that zeta = 1, no overshoot

d_cg = 0.057; % Nm -- simulates constant distrubance (ex: CG 5mm off)
% to check P control error is steady value while PI control error is 0