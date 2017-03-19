clear variables; clc;
n_iterations = 1000;

l_init = 0.9;
radius_init = 40;

rad_decay = n_iterations/log2(radius_init);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Plot decays over 'time'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = 1:1:n_iterations;
y_learn = zeros(1, n_iterations);
y_radius = zeros(1, n_iterations);

for i = 1:n_iterations
    y_learn(i) = decay_learn_rate(l_init, n_iterations, i);
    y_radius(i) = decay_radius(radius_init, i, rad_decay);
end
figure;
plot(x, y_learn, 'k--');
ylim([0 1]);
title('Learning rate over iterations');

figure;
ylim([0 radius_init]);
plot(x, y_radius, 'r--');
title('Radius over iterations');
