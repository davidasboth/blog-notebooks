function influence = calculate_influence( distance, radius )
%CALCULATE_INFLUENCE Calculate the neighbourhood influence on a neuron
%   Calculate the influence as a function of distance and radius
    influence = exp(-distance / (2* (radius^2)));
end

