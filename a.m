clc
clear all
close all

Classes = ['C1';'C2';'C3';'C4';'C5'];

imagesInfo = [dir('Training set\*.jpeg'); dir('Training set\*.jpg')];
for i = 1:length(imagesInfo)
    TrainingImages{i} = imread(strcat('Training set\' , imagesInfo(i).name));
    [r,c,dim] = size(TrainingImages{i});
    if dim == 3
        TrainingImages{i} = rgb2gray(TrainingImages{i});
    end
    TrainingLabels{i} = str2num(imagesInfo(i).name(1))+1;
    TrainingFFTs{i} = fft2(double(TrainingImages{i}));
    TrainingSignatureVector{i} = abs(TrainingFFTs{i});
    TrainingSignatureVector{i} = sort(TrainingSignatureVector{i},'descend');
    %get 5 highest spectrum magnitudes.
    TrainingSignatureVector{i} = double(TrainingSignatureVector{i}(1:5));
end

%Loading Testing Image.
ImageTest = imread('Testing set\11.png');
%apply median filter to remove Salt and Pepper noise in each channel.
RedChannel = medfilt2(ImageTest(:,:,1), [3 3]);
GreenChannel = medfilt2(ImageTest(:,:,2), [3 3]);
BlueChannel = medfilt2(ImageTest(:,:,3), [3 3]);

%Split green channel into 8 Bit Planes since red channel is blury and there is no blue channel.
cd = double(GreenChannel);
BitPlane1 = mod(cd, 2);
BitPlane2 = mod(floor(cd/2), 2);
BitPlane3 = mod(floor(cd/4), 2);
BitPlane4 = mod(floor(cd/8), 2);
BitPlane5 = mod(floor(cd/16), 2);
BitPlane6 = mod(floor(cd/32), 2);
BitPlane7 = mod(floor(cd/64), 2);
BitPlane8 = mod(floor(cd/128), 2);
%Generate new image by combining bit planes 7 and 8 since the other planes contains undesirable information.
EnhancedImage = uint8(BitPlane8*128 + BitPlane7*64);

%run fast Fourier transform fft2 algorithm for computing the discrete Fourier transform.
TestingFFT = fft2(double(EnhancedImage));
%log fft values to visualize it more brighter.
TestingFFTNormal = log(1+abs(TestingFFT));
%shift DC value to the center and put negative frequencies on the left and positive frequencies on the right.
TestingFFTShifted = log(1+abs(fftshift(TestingFFT)));
%DC value represents amplitude at 0 Hz frequency. it's also equal to MN times average value of the image f(x,y) intensities.
DC = TestingFFT(1,1);
%get 5 highest frequency domain features or spectrum magnitudes
TestingSignatureVector = abs(TestingFFT);
TestingSignatureVector = sort(TestingSignatureVector,'descend');
TestingSignatureVector = double(TestingSignatureVector(1:5));

%calculate the euclidean distances for 5 features of the testing image and 5 features for all training images.
for i = 1:length(TrainingSignatureVector)
    EuclideanDistances(i) = sqrt((TrainingSignatureVector{i}(1) - TestingSignatureVector(1,1))^2 + (TrainingSignatureVector{i}(2) - TestingSignatureVector(1,2))^2 + (TrainingSignatureVector{i}(3) - TestingSignatureVector(1,3))^2 + (TrainingSignatureVector{i}(4) - TestingSignatureVector(1,4))^2 + (TrainingSignatureVector{i}(5) - TestingSignatureVector(1,5))^2);
end

EuclideanDistances = uint32(EuclideanDistances);
%Get the minimum distance and it will be the nearest matched image in training images.
[MinDistance, Index] = min(EuclideanDistances);
ClassName = Classes(TrainingLabels{Index},:);
disp(strcat('DC Value: ', num2str(DC)));
disp(strcat('Image Class: ', ClassName));

Visualization = figure;
Visualization.Position  = [100,100,1200,400];
subplot(1,4,1), imshow(GreenChannel), title('Green Channel');
subplot(1,4,2), imshow(EnhancedImage), title('Result of Bit Planes 7+8');
subplot(1,4,3), imshow(TestingFFTNormal, []), title('FFT');
subplot(1,4,4), imshow(TestingFFTShifted, []), title('FFT Shifted');
