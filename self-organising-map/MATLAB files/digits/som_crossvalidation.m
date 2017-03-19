clear variables; clc;
start_time = clock;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Network Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(5); % for reproducibility
% setup input (m-dimensional vectors)
load('handwriting');
% reshape original vectors to 2-D matrix, and normalise
unshuffled_input = reshape(training_set,[784 size(training_set, 3)]) / 255;
% add target classes to the end of the input
unshuffled_input(785,:) = training_classes;
% random shuffle along columns
input = unshuffled_input(:,randperm(size(unshuffled_input,2)));
% number of iterations
n_iterations = 30000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Hyperparameters for cross-validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% size of the SOM (assumed to be 2-D)
network_sizes = [[1 10]; [5 5]; [10 10]; [20 20]];
% learning rate parameters
learning_rates = [0.01 0.03 0.1 0.3 0.9];
% number of folds
nFolds = 4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% range of values between which to initialise random weights
weight_range = [0 1];
% debug mode (0=OFF, 1=ON)
debug_mode = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           SOM Code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
% m is the number of input dimensions (-1 for the target class at the end)
m = size(input, 1) - 1;
% n is the number of training examples
n = size(input, 2);

% grid search for hyperparameter optimisation
n_learning_rates = size(learning_rates, 2);
n_network_sizes = size(network_sizes, 1);
fold_size = n / nFolds;

% store accuracy of each hyperparameter combination
accuracies = zeros(n_learning_rates*n_network_sizes,3);
h_counter = 1;
% loop through learning rates
for L = 1:n_learning_rates
    % set current learning rate
    init_learn_rate = learning_rates(L);
    fprintf('Training with learning rate: %d\n', init_learn_rate);
    
    % loop through different network sizes
    for N = 1:n_network_sizes
        % set current network size
        network_dimensions = network_sizes(N, :);
        fprintf('\tTraining with network size: %d-by-%d\n', ...
                network_dimensions(1), ...
                network_dimensions(2));
        % initial neighbourhood radius
        init_radius = max(network_dimensions(1), network_dimensions(2)) / 2;
        % radius decay parameter
        time_constant = n_iterations/log(init_radius);
        
        % setup network by randomly initialising weights of each vector
        a = weight_range(1);
        b = weight_range(2);
        net = (b-a) .* rand([network_dimensions m]) + a;
        
        % store accuracy per fold to then average at the end
        fold_accuracies = zeros(nFolds,1);
        for k = 1:nFolds
            fprintf('\t\tFold %d\n', k);
            % calculate folds
            if k == 1
                subTrainingSet = input(:,fold_size+1:end);
                validationSet = input(:,1:fold_size);
            elseif k == nFolds
                subTrainingSet = input(:,1:end-fold_size);
                validationSet = input(:,end-fold_size+1:end);
            else
                validationSet = input(:,((k-1)*fold_size)+1:k*fold_size);
                subTrainingSet = input;
                subTrainingSet(:,((k-1)*fold_size)+1:k*fold_size) = [];
            end
            
            trg_size = size(subTrainingSet, 2);
            
            % training
            for i = 1:n_iterations
                fprintf('\t\t\tIteration %d\n', i);
                % select a training example at random
                t = subTrainingSet(1:m, randi([1 trg_size]));

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
            % bmu_vec looks like this: x, y, c
            % where x&y are the BMU co-ordinates and c is the actual class
            bmu_vec = zeros(trg_size,3);
            class_counter = 1;
            for i = 1:trg_size
                [~, idx] = find_bmu(subTrainingSet(1:m,i), net, m, network_dimensions);
                bmu_vec(i,1:2) = idx;
                % assign class
                bmu_vec(i,3) = subTrainingSet(m+1,i);
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
                    classes = rows(:,3);
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
            end % end for (x,y) loop

            % for each test/validation example, calculate BMU
            n_test = size(validationSet, 2);
            
            test_predictions = zeros(n_test, 1);

            for i = 1:n_test
                [~, idx] = find_bmu(validationSet(1:m,i), net, m, network_dimensions);
                % 'predict' class using BMU classifier matrix
                test_predictions(i) = bmu_classes(idx(1), idx(2));
            end
            % evaluate using actual classes to get an accuracy measure
            test_classes = validationSet(m+1,:)';
            accuracy = size(test_classes(test_classes == test_predictions), 1) / n_test;
            fold_accuracies(k) = accuracy;
        end % end folds
        % get average accuracy across folds
        avg_accuracy = sum(fold_accuracies) / nFolds;
        % store accuracy against hyperparameter values
        n_neurons = network_dimensions(1) * network_dimensions(2);
        accuracies(h_counter, :) = [init_learn_rate n_neurons avg_accuracy];
        % increment array counter
        h_counter = h_counter + 1;
    end % end network_dimensions hyperparameter loop
end % end learning rates loop
fprintf('\nTraining complete!\n');
finish_time = clock;
fprintf('Started:\t%d:%d:%f\nFinished:\t%d:%d:%f\n', ...
        start_time(4),start_time(5),start_time(6), ...
        finish_time(4), finish_time(5), finish_time(6));