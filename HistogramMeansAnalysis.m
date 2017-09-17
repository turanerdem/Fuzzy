% load image
I = rgb2gray(imread('/image.jpg'));

% display image
imageSize = size(I);
numRows = imageSize(1);
numCols = imageSize(2);
figure('name', 'Original')
imshow(I)

% image matrix to observations vector
X = double(reshape(I, numRows*numCols, []));

% % kmeans
[L1, kmeansCenters] = kmeans(X,2);
kmeansClusters = reshape(L1, [numRows numCols]);
kmeansCenters = sort(kmeansCenters);
figure('name', 'Kmeans clustering')
imshow(label2rgb(kmeansClusters))

% fuzzy c-means
[centers, U] = fcm(X, 2);
[values, indexes] = max(U);
fcmClusters = reshape(indexes, [numRows numCols]);
figure('name', 'Fuzzy C-means')
imshow(label2rgb(fcmClusters)); %return;

% gaussian mixture model, normal eðrisi için
gmm = fitgmdist(X, 2);
L2 = cluster(gmm, X);
tes = gmm.mu;
sigmas = squeeze(gmm.Sigma);
gmmClusters = reshape(L2, [numRows numCols]);
figure('name', 'GMM clustering')
imshow(label2rgb(gmmClusters))

% görüntü histogramý, kmeans centroids, Fuzzy c-means, gaussian mixtures
figure('Name', 'Image histogram')
[counts, a] = imhist(I, 100);
bar(a, counts, 'b');
hold on;
plot(numRows*numCols * normpdf([0:255], tes(1), sqrt(sigmas(1))), 'g', 'LineWidth', 2);
plot(numRows*numCols * normpdf([0:255], tes(2), sqrt(sigmas(2))), 'g', 'LineWidth', 2);
xlim([0 255]);
xlabel('Intensity Value');
ylabel('Frequency of Occurrence');
plot(normpdf([0:255], tes(2), sqrt(sigmas(2))))
