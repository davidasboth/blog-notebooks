function r = decay_radius( initial_radius, i, time_constant )
%DECAY_RADIUS Decay the neighbourhood radius over time
%   Calculate decayed value based on initial radius, the number of
%   iterations, and the radius decay factor
    r = initial_radius * exp(-i/time_constant);
end

