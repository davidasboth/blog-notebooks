function render_som( network_dimensions, input_dimension, som, bmu_vec )
%RENDER_SOM Render the SOM
%   Renders the SOM using the network dimensions and weight vectors
    clf;
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
            % try drawing a star to show the 2D 'centroid'
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
        %set(gca, 'ColorOrder', bmu_vec(:,3:5));
        a = 0;
        b = 0.2;
        noise_x = (b-a) .* rand(size(bmu_vec, 1), 1) + a;
        noise_y = (b-a) .* rand(size(bmu_vec, 1), 1) + a;
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

