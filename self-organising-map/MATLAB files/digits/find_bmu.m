function [ bmu, bmu_idx ] = find_bmu( input_vector, som, input_dimension, network_dimensions )
%FIND_BMU find the indices and weight vector of the Best Matching Unit
%   finds the BMU and its lattice indices given an input vector

% by finding the neuron with the smallest Euclidean distance
    % to the training example
    idx = [1 1];
    min_dist = 1000000;
    for x = 1:network_dimensions(1)
        for y = 1:network_dimensions(2)
            % find weight vector
            w = reshape(som(x, y, :),[input_dimension 1]);
            % find the Euclidean distance (without sqrt, for speed)
            sq_dist = sum((w - input_vector) .^ 2);
            % if this is the minimum, store it
            if sq_dist < min_dist
                % store the value
                min_dist = sq_dist;
                % and the indices
                idx = [x y];
            end
        end
    end
    % return values
    bmu_idx = idx;
    bmu = reshape(som(idx(1), idx(2), :), [input_dimension 1]);

end