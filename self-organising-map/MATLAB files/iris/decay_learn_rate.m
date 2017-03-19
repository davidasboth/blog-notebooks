function l_out = decay_learn_rate(l_initial, n_iterations, i)
%DECAY_LEARN_RATE Decay the learning rate over time
%   Calculate decayed value based on initial learning rate, the number of
%   iterations, and the current iteration
    l_out = l_initial * exp(-i/n_iterations);
end