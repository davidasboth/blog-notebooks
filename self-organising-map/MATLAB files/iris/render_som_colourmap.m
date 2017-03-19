function render_som_colourmap( network_dimensions, input_dimension, som )
%RENDER_SOM_COLOURMAP Render the SOM as a colour map
%   Renders the SOM using the network dimensions and weight vectors
    clf;
    for x = 1:network_dimensions(1)
        for y = 1:network_dimensions(2)
            w = reshape(som(x, y, :),[input_dimension 1]);
            rectangle('Position', [x-0.5 y-0.5 1 1], ...
                      'FaceColor', [w(1) w(2) w(3)], ...
                      'EdgeColor', 'None');
        end
    end
    drawnow;
end