clear variables; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Network Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(4);
% setup input (m-dimensional vectors)
%input = [1, 0, 0; 0, 1, 0; 0, 0.5, 0.25; 0, 0, 1; 0, 0, 0.5; 1, 1, 0.2; 1, 0.4, 0.25; 1, 0, 1;]';
%input = iris_dataset;

load fisheriris;
input = meas';

% size of the SOM (assumed to be 2-D)
network_dimensions = [10 10];
% number of iterations
n_iterations = 1000;
% initial learning rate
init_learn_rate = 0.1;
% render the latest SOM after this many iterations
render_threshold = 100;
% range of values between which to initialise random weights
weight_range = [0 1];
% debug mode (0=OFF, 1=ON)
debug_mode = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           SOM Code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
% m is the number of input dimensions
m = size(input, 1);
% n is the number of training examples
n = size(input, 2);

% normalise the input to [0 1] interval
for d = 1:m % normalise along each dimension (row-wise)
    row = input(d,:);
    input(d,:) = (row - min(row)) / (max(row) - min(row));
end

% initial neighbourhood radius
init_radius = max(network_dimensions(1), network_dimensions(2)) / 2;
% radius decay parameter
time_constant = n_iterations/log(init_radius);

% setup network by randomly initialising weights of each vector
a = weight_range(1);
b = weight_range(2);
net = (b-a) .* rand([network_dimensions m]) + a;

% training
figure; hold on;

for i = 1:n_iterations
    fprintf('Iteration %d\n', i)
    % select a training example at random
    t = input(:, randi([1 n]));
    if debug_mode == 1
        fprintf('\tInput picked at random: (%d, %d, %d)\n', t(1), t(2), t(3));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   find the Best Matching Unit (BMU)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [bmu, bmu_idx] = find_bmu(t, net, m, network_dimensions);
    
    if debug_mode == 1
        fprintf('\tBMU is %d, %d (%.3f, %.3f, %.3f)\n', ...
            bmu_idx(1), bmu_idx(2), ...
            bmu(1), bmu(2), bmu(3));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %         Decay SOM parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % calculate the neighbourhood radius, based on the current iteration
    % and the decay
    r = decay_radius(init_radius, i, time_constant);
    if debug_mode == 1
        fprintf('\tRadius: %.4f', r);
    end
    % decay the learning rate
    l = decay_learn_rate(init_learn_rate, n_iterations, i);
    if debug_mode == 1
        fprintf('\tLearning rate: %.4f\n', l);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %      Update weight vectors (BMU and neighbours)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % loop through all the weight vectors
    for x = 1:network_dimensions(1)
        for y = 1:network_dimensions(2)
            % find weight vector
            w = reshape(net(x, y, :),[m 1]);
            % if w is within the BMU's neighbourhood
            % i.e. its distance < the current neighbourhood radius
            w_dist = sum(([x y] - bmu_idx) .^ 2);
            if w_dist < r^2
                % calculate the decayed influence on this neuron
                influence = calculate_influence(w_dist, r);
                % then update its weight according to the formula:
                %   new w = old w + (learning rate * influence * delta)
                %   where delta = input vector (t) - old w
                new_w = w + (l * influence * (t-w));
                if debug_mode == 1
                    fprintf('\tMoving %d, %d (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)\n', ...
                        x, y, ...
                        w(1), w(2), w(3), ...
                        new_w(1), new_w(2), new_w(3));
                end
                % update the vector in the net
                net(x, y, :) = reshape(new_w, [1 1 m]);
            end % end if dist < r
        end % end for y
    end % end for x
    if mod(i, render_threshold) == 0
        % display the SOM
        render_som(network_dimensions, m, net, []);
        %render_som_colourmap(network_dimensions, m, net);
    end
end % end n_iterations

% store last BMU for each input to map them to 2-D space
fprintf('Mapping inputs to BMUs after training...\n');
% loop through training examples, find BMU and assign to vector
bmu_vec = zeros(n,5);
for i = 1:n
    [~, idx] = find_bmu(input(:,i), net, m, network_dimensions);
    bmu_vec(i,1:2) = idx;
    if i<51
        bmu_vec(i,3:5) = [1 0 0];
    end
    if i>50 && i<101
        bmu_vec(i,3:5) = [0.1 0.9 0.1];
    end
    if i>100
        bmu_vec(i,3:5) = [0 0 1];
    end
end
render_som_colourmap(network_dimensions, m, net);
figure;
render_som(network_dimensions, m, net, bmu_vec);
% display the SOM
%render_som(network_dimensions, m, net);