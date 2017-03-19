clear variables; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Network Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(5);
% setup input (m-dimensional vectors)
load('handwriting');

% reshape original vectors to 2-D table, and normalise
unshuffled_input = reshape(training_set,[784 size(training_set, 3)]) / 255;

% random shuffle along columns
%input = unshuffled_input;
input = unshuffled_input(:,randperm(size(unshuffled_input,2)));

% size of the SOM (assumed to be 2-D)
network_dimensions = [20 20];
% number of iterations
n_iterations = 30000;
% initial learning rate
init_learn_rate = 0.3;
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
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   find the Best Matching Unit (BMU)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [bmu, bmu_idx] = find_bmu(t, net, m, network_dimensions);
    
    if debug_mode == 1
        fprintf('\tBMU is %d, %d\n', ...
            bmu_idx(1), bmu_idx(2));
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
                % update the vector in the net
                net(x, y, :) = reshape(new_w, [1 1 m]);
            end % end if dist < r
        end % end for y
    end % end for x
end % end n_iterations

% store last BMU for each input to map them to 2-D space
fprintf('Mapping inputs to BMUs after training...\n');
% loop through training examples, find BMU and assign to vector
% bmu_vec looks like this: x, y, r, g, b, c
% where x&y are the BMU co-ordinates, (r, g, b) is the class colour and
% c is the actual class
bmu_vec = zeros(n,6);

class_colours = [0 0 0;
                 1 0.6 0.4;
                 0.6 0.6 0.6;
                 1 0 0;
                 0 1 0;
                 0 0 1;
                 1 1 0;
                 1 0 1;
                 0 1 1;
                 0.7 0.3 0.3];
class_counter = 1;

som_training_2D = zeros(n, 3);
som_training_x_y = zeros(n, sum(network_dimensions) + 1);
som_training_xy_only = zeros(n, sum(network_dimensions) + 1);

for i = 1:n
    [~, idx] = find_bmu(unshuffled_input(:,i), net, m, network_dimensions);
    bmu_vec(i,1:2) = idx;
    %bmu_vec(i,3:5) = [1 0 0];
    
    if i > 1 && mod(i, 700) == 1
       class_counter = class_counter + 1;
    end
    % colour according to 'class' if required
    bmu_vec(i,3:5) = class_colours(class_counter, :);
    % also assign class
    bmu_vec(i,6) = class_counter-1;
    % add BMU coordinates to export data table
    som_training_2D(i,:) = [idx class_counter-1];
    % add to (X,Y) representation
    x_y_row = zeros(1, sum(network_dimensions));
    xy_only_row = zeros(1, sum(network_dimensions));
    % set x-coord to 1 in first part of vector
    x_y_row(idx(1)) = 1;
    % set y-coord to 1 in second part of vector
    x_y_row(network_dimensions(1) + idx(2)) = 1;
    % set x+y neuron to 1 for xy_only vector
    xy_only_row(sum(idx)) = 1;
    som_training_x_y(i,:) = [x_y_row class_counter-1];
    som_training_xy_only(i,:) = [xy_only_row class_counter-1];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          SOM Evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for each BMU, calculate the majority vote among classes
% 'output' will be a matrix the same size as the SOM
bmu_classes = zeros(network_dimensions);
for x = 1:network_dimensions(1)
    for y = 1:network_dimensions(2)
        % look at inputs that have this BMU
        rows = bmu_vec((bmu_vec(:,1) == x & bmu_vec(:,2) == y),:);
        classes = rows(:,6);
        % if nothing is assigned to that BMU, assign a random class
        if size(classes,1) == 0
            bmu_class = randi([0 9]);
        else
            % otherwise take a majority vote among the classes
            bmu_class = mode(classes);
        end
        % set that class to be the value of the bmu_classes matrix
        bmu_classes(x, y) = bmu_class;
    end
end

% for each test/validation example, calculate BMU
test_inputs = reshape(test_set,[784 3000]) / 255;
n_test = size(test_inputs, 2);
test_class_counter = 1;

test_predictions = zeros(n_test, 1);

for i = 1:n_test
    [~, idx] = find_bmu(test_inputs(:,i), net, m, network_dimensions);
    if i > 1 && mod(i, 300) == 1
       test_class_counter = test_class_counter + 1;
    end
    % 'predict' class using BMU classifier matrix
    test_predictions(i) = bmu_classes(idx(1), idx(2));
end
% evaluate using actual classes to get an accuracy measure
accuracy = size(test_classes(test_classes == test_predictions), 1) / n_test;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           EXPORT 2-D DATASET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% do the process (normalise, assign inputs to BMUs) for the test set
test_inputs = reshape(test_set,[784 3000]) / 255;
n_test = size(test_inputs, 2);

som_test_2D = zeros(n_test, 3);
som_test_x_y = zeros(n_test, sum(network_dimensions) + 1);
som_test_xy_only = zeros(n_test, sum(network_dimensions) + 1);

test_class_counter = 1;

for i = 1:n_test
    [~, idx] = find_bmu(test_inputs(:,i), net, m, network_dimensions);
    if i > 1 && mod(i, 300) == 1
       test_class_counter = test_class_counter + 1;
    end
    % add BMU coordinates to export data table
    som_test_2D(i,:) = [idx test_class_counter-1];
    % som_x_y has first X elements for x-coordinates
    % and the rest are Y coordinates
    % add to (X,Y) representation
    x_y_row = zeros(1, sum(network_dimensions));
    xy_only_row = zeros(1, sum(network_dimensions));
    % set x-coord to 1 in first part of vector
    x_y_row(idx(1)) = 1;
    % set y-coord to 1 in second part of vector
    x_y_row(network_dimensions(1) + idx(2)) = 1;
    % set x+y neuron to 1 for xy_only vector
    xy_only_row(sum(idx)) = 1;
    som_test_x_y(i,:) = [x_y_row class_counter-1];
    som_test_xy_only(i,:) = [xy_only_row class_counter-1];
end

% export various 2-D vectors to .mat file
save('som_output', 'som_training_2D', 'som_test_2D', ...
     'som_training_x_y', 'som_test_x_y', ...
     'som_training_xy_only', 'som_test_xy_only');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           SOM VISUALISATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create a point just for the class colours
for i = 1:size(class_colours, 1)
    rectangle('Position', [i-1.25 0.08 0.5 0.4], ...
              'FaceColor', class_colours(i, :), ...
              'EdgeColor', 'None');
end
%render_som_colourmap(network_dimensions, m, net); figure;
render_som(network_dimensions, m, net, bmu_vec);
% explicitly set axes
ax = gca;
ax.XTick = 1:1:network_dimensions(1);