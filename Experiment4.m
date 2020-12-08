clear all;close all;clc;
Result = ["Precision, Recall, TPR, FPR"];

% For fisheriris
load fisheriris
Data_fisheriris = meas;

% Numerical Conversion of Data
Class_fisheriris = zeros(size(species));
Class_fisheriris(strcmp(species,'setosa')) = 1;
Class_fisheriris(strcmp(species,'versicolor')) = 2;
Class_fisheriris(strcmp(species,'virginica')) = 3;
R_fisheriris = randperm(150);

% Training data of %80
Data_Train_fisheriris = Data_fisheriris(R_fisheriris(1:length(R_fisheriris)*0.8),:);
Class_Train_fisheriris = Class_fisheriris(R_fisheriris(1:length(R_fisheriris)*0.8),:);

% Test data of %20
Data_Test_fisheriris = Data_fisheriris(R_fisheriris((length(R_fisheriris)*0.8)+1:end),:);
Class_Test_fisheriris = Class_fisheriris(R_fisheriris((length(R_fisheriris)*0.8)+1:end),:);

%  Linear SVM
figure;
SVM_fisheriris = fitcecoc(Data_Train_fisheriris,Class_Train_fisheriris);
[Predicted_SVM_fisheriris,score_SVM_fishiriris] = predict(SVM_fisheriris,Data_Test_fisheriris);

% Linear SVM TP, TN, FN, FP Calculation
TP_SVM_fisheriris = 0;
TN_SVM_fisheriris = 0;
FN_SVM_fisheriris = 0;
FP_SVM_fisheriris = 0;
for i = 1:length(Class_Test_fisheriris)
%     TP Calculation
    if (Predicted_SVM_fisheriris(i,1) == 1 && Class_Test_fisheriris(i,1) == 1)
        TP_SVM_fisheriris = TP_SVM_fisheriris+1;
    end
%     TN Calculation
    if ((Predicted_SVM_fisheriris(i,1) == 2 || Predicted_SVM_fisheriris(i,1) == 3) && (Class_Test_fisheriris(i,1) == 2 || Class_Test_fisheriris(i,1) == 3))
        TN_SVM_fisheriris = TN_SVM_fisheriris+1;
    end
%     FN Calculation
    if (Class_Test_fisheriris(i,1) == 1 && (Predicted_SVM_fisheriris(i,1) == 2 || Predicted_SVM_fisheriris(i,1) == 3))
         FN_SVM_fisheriris = FN_SVM_fisheriris+1;
    end
%     FP Calculation
    if ((Class_Test_fisheriris(i,1) == 2 || Class_Test_fisheriris(i,1) == 3) && Predicted_SVM_fisheriris(i,1) == 1)
         FP_SVM_fisheriris = FP_SVM_fisheriris+1;
    end
end

% Calculation of Accuracy, Precision, Recall, F1-Score, TPR, FPR
Result_SVM_fisheriris = zeros(1,4);
Result_SVM_fisheriris(:,1) = (TP_SVM_fisheriris)/(TP_SVM_fisheriris+FP_SVM_fisheriris);
Result_SVM_fisheriris(:,2) = (TP_SVM_fisheriris)/(TP_SVM_fisheriris+FN_SVM_fisheriris);
Result_SVM_fisheriris(:,3) = (TP_SVM_fisheriris)/(TP_SVM_fisheriris+FN_SVM_fisheriris);
Result_SVM_fisheriris(:,4) = (FP_SVM_fisheriris)/(FP_SVM_fisheriris+TN_SVM_fisheriris);
Result
Result_SVM_fisheriris

% ROC curve
for i = 1:length(SVM_fisheriris.ClassNames)
    [X_SVM_fisheriris,Y_SVM_fisheriris,~,AUC_SVM_fisheriris] = perfcurve(Class_Test_fisheriris,score_SVM_fishiriris(:,i),SVM_fisheriris.ClassNames(i));
    plot(X_SVM_fisheriris,Y_SVM_fisheriris)
end
AUC_SVM_fisheriris
line([0 1],[0 1],'linestyle',':','color','k');
xlabel('FPR'), ylabel('TPR');
title('ROC Curve for fisheriris for SVM');

%  kNN
figure;
kNN_fisheriris = fitcknn(Data_Train_fisheriris,Class_Train_fisheriris);
[Predicted_kNN_fisheriris,score_kNN_fisheriris] = predict(kNN_fisheriris,Data_Test_fisheriris);

% kNN TP, TN, FN, FP Calculation
TP_kNN_fisheriris = 0;
TN_kNN_fisheriris = 0;
FN_kNN_fisheriris = 0;
FP_kNN_fisheriris = 0;
for i = 1:length(Class_Test_fisheriris)
%     TP Calculation
    if (Predicted_kNN_fisheriris(i,1) == 1 && Class_Test_fisheriris(i,1) == 1)
        TP_kNN_fisheriris = TP_kNN_fisheriris+1;
    end
%     TN Calculation
    if ((Predicted_kNN_fisheriris(i,1) == 2 || Predicted_kNN_fisheriris(i,1) == 3) && (Class_Test_fisheriris(i,1) == 2 || Class_Test_fisheriris(i,1) == 3))
        TN_kNN_fisheriris = TN_kNN_fisheriris+1;
    end
%     FN Calculation
    if (Class_Test_fisheriris(i,1) == 1 && (Predicted_kNN_fisheriris(i,1) == 2 || Predicted_kNN_fisheriris(i,1) == 3))
         FN_kNN_fisheriris = FN_kNN_fisheriris+1;
    end
%     FP Calculation
    if ((Class_Test_fisheriris(i,1) == 2 || Class_Test_fisheriris(i,1) == 3) && Predicted_kNN_fisheriris(i,1) == 1)
         FP_kNN_fisheriris = FP_kNN_fisheriris+1;
    end
end

% Calculation of Accuracy, Precision, Recall, F1-Score, TPR, FPR
Result_kNN_fisheriris = zeros(1,4);
Result_kNN_fisheriris(:,1) = (TP_kNN_fisheriris)/(TP_kNN_fisheriris+FP_kNN_fisheriris);
Result_kNN_fisheriris(:,2) = (TP_kNN_fisheriris)/(TP_kNN_fisheriris+FN_kNN_fisheriris);
Result_kNN_fisheriris(:,3) = (TP_kNN_fisheriris)/(TP_kNN_fisheriris+FN_kNN_fisheriris);
Result_kNN_fisheriris(:,4) = (FP_kNN_fisheriris)/(FP_kNN_fisheriris+TN_kNN_fisheriris);
Result
Result_kNN_fisheriris

% ROC curve
for i = 1:length(kNN_fisheriris.ClassNames)
    [X_kNN_fisheriris,Y_kNN_fisheriris,~,AUC_kNN_fisheriris] = perfcurve(Class_Test_fisheriris,score_kNN_fisheriris(:,i),kNN_fisheriris.ClassNames(i));
    plot(X_kNN_fisheriris,Y_kNN_fisheriris)
end
AUC_kNN_fisheriris
line([0 1],[0 1],'linestyle',':','color','k');
xlabel('FPR'), ylabel('TPR');
title('ROC Curve for fisheriris for kNN');

%  Decision Tree
figure;
DTree_fisheriris = fitctree(Data_Train_fisheriris,Class_Train_fisheriris);
[Predicted_DTree_fisheriris,score_DTree_fisheriris] = predict(DTree_fisheriris,Data_Test_fisheriris);

% Decision Tree TP, TN, FN, FP Calculation
TP_DTree_fisheriris = 0;
TN_DTree_fisheriris = 0;
FN_DTree_fisheriris = 0;
FP_DTree_fisheriris = 0;
for i = 1:length(Class_Test_fisheriris)
%     TP Calculation
    if (Predicted_DTree_fisheriris(i,1) == 1 && Class_Test_fisheriris(i,1) == 1)
        TP_DTree_fisheriris = TP_DTree_fisheriris+1;
    end
%     TN Calculation
    if ((Predicted_DTree_fisheriris(i,1) == 2 || Predicted_DTree_fisheriris(i,1) == 3) && (Class_Test_fisheriris(i,1) == 2 || Class_Test_fisheriris(i,1) == 3))
        TN_DTree_fisheriris = TN_DTree_fisheriris+1;
    end
%     FN Calculation
    if (Class_Test_fisheriris(i,1) == 1 && (Predicted_DTree_fisheriris(i,1) == 2 || Predicted_DTree_fisheriris(i,1) == 3))
         FN_DTree_fisheriris = FN_DTree_fisheriris+1;
    end
%     FP Calculation
    if ((Class_Test_fisheriris(i,1) == 2 || Class_Test_fisheriris(i,1) == 3) && Predicted_DTree_fisheriris(i,1) == 1)
         FP_DTree_fisheriris = FP_DTree_fisheriris+1;
    end
end

% Calculation of Accuracy, Precision, Recall, F1-Score, TPR, FPR
Result_DTree_fisheriris = zeros(1,4);
Result_DTree_fisheriris(:,1) = (TP_DTree_fisheriris)/(TP_DTree_fisheriris+FP_DTree_fisheriris);
Result_DTree_fisheriris(:,2) = (TP_DTree_fisheriris)/(TP_DTree_fisheriris+FN_DTree_fisheriris);
Result_DTree_fisheriris(:,3) = (TP_DTree_fisheriris)/(TP_DTree_fisheriris+FN_DTree_fisheriris);
Result_DTree_fisheriris(:,4) = (FP_DTree_fisheriris)/(FP_DTree_fisheriris+TN_DTree_fisheriris);
Result
Result_DTree_fisheriris

% ROC curve
for i = 1:length(DTree_fisheriris.ClassNames)
    [X_DTree_fisheriris,Y_DTree_fisheriris,~,AUC_DTree_fisheriris] = perfcurve(Class_Test_fisheriris,score_DTree_fisheriris(:,i),DTree_fisheriris.ClassNames(i));
    plot(X_DTree_fisheriris,Y_DTree_fisheriris)
end
AUC_DTree_fisheriris
line([0 1],[0 1],'linestyle',':','color','k');
xlabel('FPR'), ylabel('TPR');
title('ROC Curve for fisheriris for Decision Tree');
% 
% 
% 
% 
% 
% 
% For ionosphere
load ionosphere
Data_ionosphere = X;

% Numerical Conversion of Data
Class_ionosphere = zeros(size(Y));
Class_ionosphere(strcmp(Y,'g')) = 1;
Class_ionosphere(strcmp(Y,'b')) = 2;
R_ionosphere = randperm(length(Y));

% Training data of %80
Data_Train_ionosphere = Data_ionosphere(R_ionosphere(1:fix(length(R_ionosphere)*0.8)),:);
Class_Train_ionosphere = Class_ionosphere(R_ionosphere(1:fix(length(R_ionosphere)*0.8)),:);

% Test data of %20
Data_Test_ionosphere = Data_ionosphere(R_ionosphere(fix(length(R_ionosphere)*0.8)+1:end),:);
Class_Test_ionosphere = Class_ionosphere(R_ionosphere(fix(length(R_ionosphere)*0.8)+1:end),:);

%  Linear SVM
figure;
SVM_ionosphere = fitcsvm(Data_Train_ionosphere,Class_Train_ionosphere);
[Predicted_SVM_ionosphere,score_SVM_ionosphere] = predict(SVM_ionosphere,Data_Test_ionosphere);

% Linear SVM TP, TN, FN, FP Calculation
TP_SVM_ionosphere = 0;
TN_SVM_ionosphere = 0;
FN_SVM_ionosphere = 0;
FP_SVM_ionosphere = 0;
for i = 1:length(Class_Test_ionosphere)
%     TP Calculation
    if (Predicted_SVM_ionosphere(i,1) == 1 && Class_Test_ionosphere(i,1) == 1)
        TP_SVM_ionosphere = TP_SVM_ionosphere+1;
    end
%     TN Calculation
    if (Predicted_SVM_ionosphere(i,1) == 2 && Class_Test_ionosphere(i,1) == 2)
        TN_SVM_ionosphere = TN_SVM_ionosphere+1;
    end
%     FN Calculation
    if (Class_Test_ionosphere(i,1) == 1 && Predicted_SVM_ionosphere(i,1) == 2)
         FN_SVM_ionosphere = FN_SVM_ionosphere+1;
    end
%     FP Calculation
    if (Class_Test_ionosphere(i,1) == 2 || Predicted_SVM_ionosphere(i,1) == 1)
         FP_SVM_ionosphere = FP_SVM_ionosphere+1;
    end
end

% Calculation of Accuracy, Precision, Recall, F1-Score, TPR, FPR
Result_SVM_ionosphere = zeros(1,4);
Result_SVM_ionosphere(:,1) = (TP_SVM_ionosphere)/(TP_SVM_ionosphere+FP_SVM_ionosphere);
Result_SVM_ionosphere(:,2) = (TP_SVM_ionosphere)/(TP_SVM_ionosphere+FN_SVM_ionosphere);
Result_SVM_ionosphere(:,3) = (TP_SVM_ionosphere)/(TP_SVM_ionosphere+FN_SVM_ionosphere);
Result_SVM_ionosphere(:,4) = (FP_SVM_ionosphere)/(FP_SVM_ionosphere+TN_SVM_ionosphere);
Result
Result_SVM_ionosphere

% ROC curve
for i = 1:length(SVM_ionosphere.ClassNames)
    [X_SVM_ionosphere,Y_SVM_ionosphere,~,AUC_SVM_ionosphere] = perfcurve(Class_Test_ionosphere,score_SVM_ionosphere(:,i),SVM_ionosphere.ClassNames(i));
    plot(X_SVM_ionosphere,Y_SVM_ionosphere)
end
AUC_SVM_ionosphere
line([0 1],[0 1],'linestyle',':','color','k');
xlabel('FPR'), ylabel('TPR');
title('ROC Curve for ionosphere for SVM');

%  kNN
figure;
kNN_ionosphere = fitcknn(Data_Train_ionosphere,Class_Train_ionosphere);
[Predicted_kNN_ionosphere,score_kNN_ionosphere] = predict(kNN_ionosphere,Data_Test_ionosphere);

% kNN TP, TN, FN, FP Calculation
TP_kNN_ionosphere = 0;
TN_kNN_ionosphere = 0;
FN_kNN_ionosphere = 0;
FP_kNN_ionosphere = 0;
for i = 1:length(Class_Test_ionosphere)
%     TP Calculation
    if (Predicted_kNN_ionosphere(i,1) == 1 && Class_Test_ionosphere(i,1) == 1)
        TP_kNN_ionosphere = TP_kNN_ionosphere+1;
    end
%     TN Calculation
    if (Predicted_kNN_ionosphere(i,1) == 2 && Class_Test_ionosphere(i,1) == 2)
        TN_kNN_ionosphere = TN_kNN_ionosphere+1;
    end
%     FN Calculation
    if (Class_Test_ionosphere(i,1) == 1 && Predicted_kNN_ionosphere(i,1) == 2)
         FN_kNN_ionosphere = FN_kNN_ionosphere+1;
    end
%     FP Calculation
    if (Class_Test_ionosphere(i,1) == 2 && Predicted_kNN_ionosphere(i,1) == 1)
         FP_kNN_ionosphere = FP_kNN_ionosphere+1;
    end
end

% Calculation of Accuracy, Precision, Recall, F1-Score, TPR, FPR
Result_kNN_ionosphere = zeros(1,4);
Result_kNN_ionosphere(:,1) = (TP_kNN_ionosphere)/(TP_kNN_ionosphere+FP_kNN_ionosphere);
Result_kNN_ionosphere(:,2) = (TP_kNN_ionosphere)/(TP_kNN_ionosphere+FN_kNN_ionosphere);
Result_kNN_ionosphere(:,3) = (TP_kNN_ionosphere)/(TP_kNN_ionosphere+FN_kNN_ionosphere);
Result_kNN_ionosphere(:,4) = (FP_kNN_ionosphere)/(FP_kNN_ionosphere+TN_kNN_ionosphere);
Result
Result_kNN_ionosphere

% ROC curve
for i = 1:length(kNN_ionosphere.ClassNames)
    [X_kNN_ionosphere,Y_kNN_ionosphere,~,AUC_kNN_ionosphere] = perfcurve(Class_Test_ionosphere,score_kNN_ionosphere(:,i),kNN_ionosphere.ClassNames(i));
    plot(X_kNN_ionosphere,Y_kNN_ionosphere)
end
AUC_kNN_ionosphere
line([0 1],[0 1],'linestyle',':','color','k');
xlabel('FPR'), ylabel('TPR');
title('ROC Curve for ionosphere for kNN');

%  Decision Tree
figure;
DTree_ionosphere = fitctree(Data_Train_ionosphere,Class_Train_ionosphere);
[Predicted_DTree_ionosphere,score_DTree_ionosphere] = predict(DTree_ionosphere,Data_Test_ionosphere);

% Decision Tree TP, TN, FN, FP Calculation
TP_DTree_ionosphere = 0;
TN_DTree_ionosphere = 0;
FN_DTree_ionosphere = 0;
FP_DTree_ionosphere = 0;
for i = 1:length(Class_Test_ionosphere)
%     TP Calculation
    if (Predicted_DTree_ionosphere(i,1) == 1 && Class_Test_ionosphere(i,1) == 1)
        TP_DTree_ionosphere = TP_DTree_ionosphere+1;
    end
%     TN Calculation
    if (Predicted_DTree_ionosphere(i,1) == 2 && Class_Test_ionosphere(i,1) == 2)
        TN_DTree_ionosphere = TN_DTree_ionosphere+1;
    end
%     FN Calculation
    if (Class_Test_ionosphere(i,1) == 1 && Predicted_DTree_ionosphere(i,1) == 2)
         FN_DTree_ionosphere = FN_DTree_ionosphere+1;
    end
%     FP Calculation
    if (Class_Test_ionosphere(i,1) == 2 && Predicted_DTree_ionosphere(i,1) == 1)
         FP_DTree_ionosphere = FP_DTree_ionosphere+1;
    end
end

% Calculation of Accuracy, Precision, Recall, F1-Score, TPR, FPR
Result_DTree_ionosphere = zeros(1,4);
Result_DTree_ionosphere(:,1) = (TP_DTree_ionosphere)/(TP_DTree_ionosphere+FP_DTree_ionosphere);
Result_DTree_ionosphere(:,2) = (TP_DTree_ionosphere)/(TP_DTree_ionosphere+FN_DTree_ionosphere);
Result_DTree_ionosphere(:,3) = (TP_DTree_ionosphere)/(TP_DTree_ionosphere+FN_DTree_ionosphere);
Result_DTree_ionosphere(:,4) = (FP_DTree_ionosphere)/(FP_DTree_ionosphere+TN_DTree_ionosphere);
Result
Result_DTree_ionosphere

% ROC curve
for i = 1:length(DTree_ionosphere.ClassNames)
    [X_DTree_ionosphere,Y_DTree_ionosphere,~,AUC_DTree_ionosphere] = perfcurve(Class_Test_ionosphere,score_DTree_ionosphere(:,i),DTree_ionosphere.ClassNames(i));
    plot(X_DTree_ionosphere,Y_DTree_ionosphere)
end
AUC_DTree_ionosphere
line([0 1],[0 1],'linestyle',':','color','k');
xlabel('FPR'), ylabel('TPR');
title('ROC Curve for ionosphere for Decision Tree');