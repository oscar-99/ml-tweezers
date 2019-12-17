function job(slurm_index, slurm_total)

addpath('/data/uqilento/ott');
base = '.';

% Load the beams, Tmatrix and positions
load([base, '/input.mat']);

numpts = size(positions, 2);
%numpts = 100;		% For testing

% Split the data (5 DOF)
numpos = floor(numpts./slurm_total);
positions = positions(:, (1:numpos)+(slurm_index-1)*numpos);
radius = radius(:, (1:numpos)+(slurm_index-1)*numpos);
index_particle = index_particle(:, (1:numpos)+(slurm_index-1)*numpos);

% Allocate output
forces = zeros(3, numpos);

tic

for ii = 1:numpos

  % Generate T-matrix from radius, index, meta
  Tmatrix = ott.Tmatrix.simple('sphere', radius(ii), ...
    'wavelength0', meta.wavelength0, ...
    'index_medium', meta.index_medium, ...
    'index_particle', index_particle(ii));

  % Calculate the optical force [Q]
  forces(:, ii) = ott.forcetorque(beam, Tmatrix, 'position', positions(:, ii));

end

% Change forces to [N]
forces = forces ./ beam.speed;

time = toc();

oname = [base, '/output/s', num2str(slurm_index), '.mat'];
save(oname, 'radius', 'index_particle', 'positions', ...
    'forces', 'time', 'meta');

