function render_som( network_dimensions, input_dimension, som, bmu_vec )
%RENDER_SOM Render the SOM
%   Renders the SOM using the network dimensions and weight vectors
    %clf;
    % vectors to store centroids
    x_vec = zeros(1, network_dimensions(1));
    y_vec = zeros(1, network_dimensions(2));
    % vectors to store lattice nodes
    x_coords = zeros(1, network_dimensions(1));
    y_coords = zeros(1, network_dimensions(2));
    counter = 1;
    for x = 1:network_dimensions(1)
        for y = 1:network_dimensions(2)
            w = reshape(som(x, y, :),[input_dimension 1]);
            % store the 2D 'centroid', i.e. the first 2 components of the
            % weight vector
            x_vec(counter) = w(1);
            y_vec(counter) = w(2);
            % store the lattice points
            x_coords(counter) = x;
            y_coords(counter) = y;
            counter = counter + 1;
        end
    end
    % plot the underlying nodes
    plot(x_coords, y_coords, 'bo'); hold on;
    % plot the bmus
    if size(bmu_vec, 1) > 0
        % noise parameters in x and y direction when showing colours
        a_x = -0.4;
        a_y = -0.4;
        b_x = 0.4;
        b_y = 0.4;
        noise_x = (b_x-a_x) .* rand(size(bmu_vec, 1), 1) + a_x;
        noise_y = (b_y-a_y) .* rand(size(bmu_vec, 1), 1) + a_y;
        scatter(bmu_vec(:,1) + noise_x, ...
                bmu_vec(:,2) + noise_y, ...
                18, ...
                bmu_vec(:,3:5), ...
                'filled');
    end
    %plot(x_vec, y_vec, 'k*');
    xlim([0 max(x_coords)+1]);
    ylim([0 max(y_coords)+1]);
    drawnow;
end

