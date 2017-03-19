clear variables; clc;
% for reproducibility
rng(8);

% set percentage of test set
test_set_pct = 30;

% data vector, 10000 instances of 28x28 vectors
img_array = zeros(28,28,10000);
% instantiate proportional training & test sets
training_set = [];
training_classes = [];

test_set = [];
test_classes = [];

for f=0:9
    fname = strcat('data/data', num2str(f));
    fid = fopen(fname, 'r');
    for i = (f*1000) + 1 : (f+1) * 1000
        img_array(:,:,i) = fread(fid,[28 28],'uchar');
    end
    % sample for training & test data
    % sample test data indices between the current 1000 rows, along the 3rd
    % dimension, WITHOUT replacement
    [~, idx] = datasample(img_array(:,:,(f*1000) + 1 : (f+1) * 1000), test_set_pct*10, 3, 'Replace', false);
    % add sample onto test set (add f*1000 to idx because for each
    % subset, we are sampling between 1 and 1000 for the 9th iteration,
    % these values need to be between 8001 and 9000)
    test_set = cat(3, test_set, img_array(:,:,idx + (f*1000)));
    % assign classes to test set
    test_classes = cat(1,test_classes,repmat(f, test_set_pct*10, 1));
    % get the inverse of the sampled indices
    trg_idx = setdiff(1:1000, idx);
    training_set = cat(3, training_set, img_array(:,:,trg_idx + (f*1000)));
    % assign classes to training set
    training_classes = cat(1,training_classes,repmat(f, (100-test_set_pct)*10, 1));
end

n_rows = size(training_set,3);

train_to_shuffle = zeros(n_rows,785);

for t = 1:n_rows
    %fprintf('\tEpisode %d, row %d\n', e, t);
    % pick up the t'th row from the training_set
    train_to_shuffle(t,1:784) = reshape(training_set(:,:,t),1,784);
end
train_to_shuffle(:,785)=training_classes;
% shuffle rows of matrix

shuffledTrain = train_to_shuffle(randperm(size(train_to_shuffle,1)),:);

n_rows_test = size(test_set,3);

test_to_shuffle = zeros(n_rows_test,785);

for t = 1:n_rows_test
    %fprintf('\tEpisode %d, row %d\n', e, t);
    % pick up the t'th row from the training_set
    test_to_shuffle(t,1:784) = reshape(test_set(:,:,t),1,784);
end
test_to_shuffle(:,785)=test_classes;
% shuffle rows of matrix
% not strictly necessary for held out test, done for consistency
shuffledTest = test_to_shuffle(randperm(size(test_to_shuffle,1)),:);


save('handwritingv2', 'shuffledTrain', 'shuffledTest');

% Uncomment these for verification

% Verify training set
%i = randi([1 size(training_set,3)],1,1);
%imshow(training_set(:,:,i));
%fprintf('Image is class %d\n', training_classes(i));

% Verify test set
%i = randi([1 size(test_set,3)],1,1);
%imshow(test_set(:,:,i));
%fprintf('Image is class %d\n', test_classes(i));
