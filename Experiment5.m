clear all;close all;clc;

% For fisheriris
load fisheriris
Data_fisheriris = meas;

% Numerical Conversion of Data
Class_fisheriris = zeros(size(species));
Class_fisheriris(strcmp(species,'setosa')) = 1;
Class_fisheriris(strcmp(species,'versicolor')) = 2;
Class_fisheriris(strcmp(species,'virginica')) = 3;
R_fisheriris = randperm(length(species))';

% Training data of %80
Data_Train_fisheriris = Data_fisheriris(R_fisheriris(1:length(R_fisheriris)*0.8),:);
Class_Train_fisheriris = Class_fisheriris(R_fisheriris(1:length(R_fisheriris)*0.8),:);

% Test data of %20
Data_Test_fisheriris = Data_fisheriris(R_fisheriris((length(R_fisheriris)*0.8)+1:end),:);
Class_Test_fisheriris = Class_fisheriris(R_fisheriris((length(R_fisheriris)*0.8)+1:end),:);

% k-Means
for k = 1:6
    kMeans_fisheriris = kmeans(Data_Train_fisheriris,k);
%     predicttt = predict(kMeans_fisheriris,Data_Test_fisheriris);
end