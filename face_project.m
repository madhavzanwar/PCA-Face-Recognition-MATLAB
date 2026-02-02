clc;
clear;
close all;

A = [];

for person = 1:32
    folder = ['s' num2str(person)];

    for imgNum = 1:6
        img = imread([folder '/' num2str(imgNum) '.pgm']);
        img = double(img);
        vec = img(:);

        A = [A vec];
    end
end

disp(size(A));

%compute mean face
meanface = mean(A, 2);

%show mean face
meanImg = reshape(meanface, 112, 92);
figure;
imshow(uint8(meanImg));
title('Mean Face');

% Subtract mean face
A_centered = A - meanface;

disp(size(A_centered));

% PCA trick matrix
L = A_centered' * A_centered;

% Eigen decomposition
[V, D] = eig(L);

% Sort eigenvalues descending
eigenvalues = diag(D);
[~, idx] = sort(eigenvalues, 'descend');

V = V(:, idx);

% Compute eigenfaces
eigenfaces = A_centered * V;

%normalize eigen faces
for i = 1:size(eigenfaces,2)
    eigenfaces(:,i) = eigenfaces(:,i) / norm(eigenfaces(:,i));
end

figure;
for i = 1:25
    subplot(5,5,i);
    ef = reshape(eigenfaces(:,i), 112, 92);
     ef = ef - min(ef(:));
     ef = ef / max(ef(:));
    imshow(ef);

    title(['EF ' num2str(i)]);
end
sgtitle('Top 25 Eigenfaces');

% Project all training faces
trainWeights = eigenfaces' * A_centered;

disp(size(trainWeights));


% Load test image
testImg = imread('s1/7.pgm');
testImg = double(testImg);
testVec = testImg(:);

% Center using mean face
testCentered = testVec - meanface;

% Project into PCA space
testWeights = eigenfaces' * testCentered;

% Compare with all training faces
diffs = vecnorm(trainWeights - testWeights);

% Find best match
[~, idx] = min(diffs);

disp(['Matched training image index: ' num2str(idx)]);

% Identify person
predictedPerson = ceil(idx/6);
disp(['Predicted person: s' num2str(predictedPerson)]);


correct = 0;
total = 0;

for person = 1:32
    for imgNum = 7:10
        
        % Load test image
        testImg = imread(['s' num2str(person) '/' num2str(imgNum) '.pgm']);
        testImg = double(testImg);
        testVec = testImg(:);

        % Center
        testCentered = testVec - meanface;

        % Project
        testWeights = eigenfaces' * testCentered;

        % Compare
        diffs = vecnorm(trainWeights - testWeights);
        [~, idx] = min(diffs);

        predictedPerson = ceil(idx/6);

        if predictedPerson == person
            correct = correct + 1;
        end

        total = total + 1;
    end
end

accuracy = (correct/total)*100;
disp(['Final Accuracy: ' num2str(accuracy) '%']);


k = 20; 

sample = A_centered(:,1);

recon = eigenfaces(:,1:k) * ...
        (eigenfaces(:,1:k)' * sample);

recon = recon + meanface;

figure;

subplot(1,2,1);
imshow(reshape(A(:,1),112,92),[]);
title('Original');

subplot(1,2,2);
imshow(reshape(recon,112,92),[]);
title(['Reconstructed k=' num2str(k)]);
