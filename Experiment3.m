clear all;close all;clc;

% For fisheriris
load fisheriris
Data_fisheriris = meas;

% Numerical Conversion of Data
Class_fisheriris = zeros(size(species));
Class_fisheriris(strcmp(species,'setosa')) = 1;
Class_fisheriris(strcmp(species,'versicolor')) = 2;
Class_fisheriris(strcmp(species,'virginica')) = 3;

% k-Fold Crossvalidation Calculation
for k = 5:10  % k=5,6,7,8,9,10
    INDICES_fisheriris = crossvalind('Kfold',Class_fisheriris,k);
    for j = 1:k
        test = (INDICES_fisheriris == j);
        train = ~test ;
        
%         Train Data
        Data_Train_fisheriris = Data_fisheriris(train,:);
        Class_Train_fisheriris = Class_fisheriris(train,:);
        
%         Test Data
        Data_Test_fisheriris = Data_fisheriris(test,:);
        Class_Test_fisheriris = Class_fisheriris(test,:);
        
%         Linear SVM
        SVM_fisheriris = fitcecoc(Data_Train_fisheriris,Class_Train_fisheriris);
        Predicted_SVM_fisheriris = predict(SVM_fisheriris,Data_Test_fisheriris);
        
%         Linear SVM TP, TN, FN, FP Calculation
        TP_SVM_fisheriris = 0;
        TN_SVM_fisheriris = 0;
        FN_SVM_fisheriris = 0;
        FP_SVM_fisheriris = 0; 
        for i = 1:length(Class_Test_fisheriris)
            if (Predicted_SVM_fisheriris(i,1) == 1 && Class_Test_fisheriris(i,1) == 1)
                TP_SVM_fisheriris = TP_SVM_fisheriris+1;
            end
            if ((Predicted_SVM_fisheriris(i,1) == 2 || Predicted_SVM_fisheriris(i,1) == 3) && (Class_Test_fisheriris(i,1) == 2 || Class_Test_fisheriris(i,1) == 3))
                TN_SVM_fisheriris = TN_SVM_fisheriris+1;
            end
            if ((Predicted_SVM_fisheriris(i,1) == 2 || Predicted_SVM_fisheriris(i,1) == 3) && Class_Test_fisheriris(i,1) == 1)
                FN_SVM_fisheriris = FN_SVM_fisheriris+1;
            end
            if (Predicted_SVM_fisheriris(i,1) == 1 && (Class_Test_fisheriris(i,1) == 2 || Class_Test_fisheriris(i,1) == 3))
                FP_SVM_fisheriris = FP_SVM_fisheriris+1;
            end
        end

%         kNN
        kNN_fisheriris = fitcknn(Data_Train_fisheriris,Class_Train_fisheriris);
        Predicted_kNN_fisheriris = predict(kNN_fisheriris,Data_Test_fisheriris);
        
%         kNN TP, TN, FN, FP Calculation
        TP_kNN_fisheriris = 0;
        TN_kNN_fisheriris = 0;
        FN_kNN_fisheriris = 0;
        FP_kNN_fisheriris = 0; 
        for i = 1:length(Class_Test_fisheriris)
            if (Predicted_kNN_fisheriris(i,1) == 1 && Class_Test_fisheriris(i,1) == 1)
                TP_kNN_fisheriris = TP_kNN_fisheriris+1;
            end
            if ((Predicted_kNN_fisheriris(i,1) == 2 || Predicted_kNN_fisheriris(i,1) == 3) && (Class_Test_fisheriris(i,1) == 2 || Class_Test_fisheriris(i,1) == 3))
                TN_kNN_fisheriris = TN_kNN_fisheriris+1;
            end
            if ((Predicted_kNN_fisheriris(i,1) == 2 || Predicted_kNN_fisheriris(i,1) == 3) && Class_Test_fisheriris(i,1) == 1)
                FN_kNN_fisheriris = FN_kNN_fisheriris+1;
            end
            if (Predicted_kNN_fisheriris(i,1) == 1 && (Class_Test_fisheriris(i,1) == 2 || Class_Test_fisheriris(i,1) == 3))
                FP_kNN_fisheriris = FP_kNN_fisheriris+1;
            end
        end

%         Decision Tree
        DTree_fisheriris = fitctree(Data_Train_fisheriris,Class_Train_fisheriris);
        Predicted_DTree_fisheriris = predict(DTree_fisheriris,Data_Test_fisheriris);
        
%         Decision Tree TP, TN, FN, FP Calculation
        TP_DTree_fisheriris = 0;
        TN_DTree_fisheriris = 0;
        FN_DTree_fisheriris = 0;
        FP_DTree_fisheriris = 0; 
        for i = 1:length(Class_Test_fisheriris)
            if (Predicted_DTree_fisheriris(i,1) == 1 && Class_Test_fisheriris(i,1) == 1)
                TP_DTree_fisheriris = TP_DTree_fisheriris+1;
            end
            if ((Predicted_DTree_fisheriris(i,1) == 2 || Predicted_DTree_fisheriris(i,1) == 3) && (Class_Test_fisheriris(i,1) == 2 || Class_Test_fisheriris(i,1) == 3))
                TN_DTree_fisheriris = TN_DTree_fisheriris+1;
            end
            if ((Predicted_DTree_fisheriris(i,1) == 2 || Predicted_DTree_fisheriris(i,1) == 3) && Class_Test_fisheriris(i,1) == 1)
                FN_DTree_fisheriris = FN_DTree_fisheriris+1;
            end
            if (Predicted_DTree_fisheriris(i,1) == 1 && (Class_Test_fisheriris(i,1) == 2 || Class_Test_fisheriris(i,1) == 3))
                FP_DTree_fisheriris = FP_DTree_fisheriris+1;
            end
        end
        
%         Precision, Recall Calculation
        Precison_SVM_fisheriris(1,j) =  (TP_SVM_fisheriris)/(TP_SVM_fisheriris+FP_SVM_fisheriris);
        Recall_SVM_fisheriris(1,j) = (TP_SVM_fisheriris)/(TP_SVM_fisheriris+FN_SVM_fisheriris);
        Precison_kNN_fisheriris(1,j) =  (TP_kNN_fisheriris)/(TP_kNN_fisheriris+FP_kNN_fisheriris);
        Recall_kNN_fisheriris(1,j) = (TP_kNN_fisheriris)/(TP_kNN_fisheriris+FN_kNN_fisheriris);
        Precison_DTree_fisheriris(1,j) =  (TP_DTree_fisheriris)/(TP_DTree_fisheriris+FP_DTree_fisheriris);
        Recall_DTree_fisheriris(1,j) = (TP_DTree_fisheriris)/(TP_DTree_fisheriris+FN_DTree_fisheriris);
    end
    Mean_Precison_SVM_fisheriris(1,k-4) = mean(Precison_SVM_fisheriris);
    Mean_Recall_SVM_fisheriris(1,k-4) = mean(Recall_SVM_fisheriris);
    Mean_Precison_kNN_fisheriris(1,k-4) = mean(Precison_kNN_fisheriris);
    Mean_Recall_kNN_fisheriris(1,k-4) = mean(Recall_kNN_fisheriris);
    Mean_Precison_DTree_fisheriris(1,k-4) = mean(Precison_DTree_fisheriris);
    Mean_Recall_DTree_fisheriris(1,k-4) = mean(Recall_DTree_fisheriris);
end

% plot For Precision and Recall
k = 5:10;
figure;plot(k,Mean_Precison_SVM_fisheriris);
title('precision Graph For fisheriris SVM');xlabel('k Value');ylabel('precision Value');
figure;plot(k,Mean_Recall_SVM_fisheriris);
title('Recall Graph For fisheriris SVM');xlabel('k Value');ylabel('Recall Value');
figure;plot(k,Mean_Precison_kNN_fisheriris);
title('precision Graph For fisheriris kNN');xlabel('k Value');ylabel('precision Value');
figure;plot(k,Mean_Recall_kNN_fisheriris);
title('Recall Graph For fisheriris kNN');xlabel('k Value');ylabel('Recall Value');
figure;plot(k,Mean_Precison_DTree_fisheriris);
title('precision Graph For fisheriris Decision Tree');xlabel('k Value');ylabel('precision Value');
figure;plot(k,Mean_Recall_DTree_fisheriris);
title('Recall Graph For fisheriris Decision Tree');xlabel('k Value');ylabel('Recall Value');
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

% k-Fold Crossvalidation Calculation
for k = 5:10  % k=5,6,7,8,9,10
    INDICES_ionosphere = crossvalind('Kfold',Class_ionosphere,k);
    for j = 1:k
        test = (INDICES_ionosphere == j);
        train = ~test ;
        
%         Train Data
        Data_Train_ionosphere = Data_ionosphere(train,:);
        Class_Train_ionosphere = Class_ionosphere(train,:);
        
%         Test Data
        Data_Test_ionosphere = Data_ionosphere(test,:);
        Class_Test_ionosphere = Class_ionosphere(test,:);
        
%         Linear SVM
        SVM_ionosphere = fitcsvm(Data_Train_ionosphere,Class_Train_ionosphere);
        Predicted_SVM_ionosphere = predict(SVM_ionosphere,Data_Test_ionosphere);
        
%         Linear SVM TP, TN, FN, FP Calculation
        TP_SVM_ionosphere = 0;
        TN_SVM_ionosphere = 0;
        FN_SVM_ionosphere = 0;
        FP_SVM_ionosphere = 0; 
        for i = 1:length(Class_Test_ionosphere)
            if (Predicted_SVM_ionosphere(i,1) == 1 && Class_Test_ionosphere(i,1) == 1)
                TP_SVM_ionosphere = TP_SVM_ionosphere+1;
            end
            if (Predicted_SVM_ionosphere(i,1) == 2 && Class_Test_ionosphere(i,1) == 2)
                TN_SVM_ionosphere = TN_SVM_ionosphere+1;
            end
            if (Predicted_SVM_ionosphere(i,1) == 2 && Class_Test_ionosphere(i,1) == 1)
                FN_SVM_ionosphere = FN_SVM_ionosphere+1;
            end
            if (Predicted_SVM_ionosphere(i,1) == 1 && Class_Test_ionosphere(i,1) == 2)
                FP_SVM_ionosphere = FP_SVM_ionosphere+1;
            end
        end

%         kNN
        kNN_ionosphere = fitcknn(Data_Train_ionosphere,Class_Train_ionosphere);
        Predicted_kNN_ionosphere = predict(kNN_ionosphere,Data_Test_ionosphere);
        
%         kNN TP, TN, FN, FP Calculation
        TP_kNN_ionosphere = 0;
        TN_kNN_ionosphere = 0;
        FN_kNN_ionosphere = 0;
        FP_kNN_ionosphere = 0;
        for i = 1:length(Class_Test_ionosphere)
            if (Predicted_kNN_ionosphere(i,1) == 1 && Class_Test_ionosphere(i,1) == 1)
                TP_kNN_ionosphere = TP_kNN_ionosphere+1;
            end
            if (Predicted_kNN_ionosphere(i,1) == 2 && Class_Test_ionosphere(i,1) == 2)
                TN_kNN_ionosphere = TN_kNN_ionosphere+1;
            end
            if (Predicted_kNN_ionosphere(i,1) == 2 && Class_Test_ionosphere(i,1) == 1)
                FN_kNN_ionosphere = FN_kNN_ionosphere+1;
            end
            if (Predicted_kNN_ionosphere(i,1) == 1 && Class_Test_ionosphere(i,1) == 2)
                FP_kNN_ionosphere = FP_kNN_ionosphere+1;
            end
        end

%         Decision Tree
        DTree_ionosphere = fitctree(Data_Train_ionosphere,Class_Train_ionosphere);
        Predicted_DTree_ionosphere = predict(DTree_ionosphere,Data_Test_ionosphere);
        
%         Decision Tree TP, TN, FN, FP Calculation
        TP_DTree_ionosphere = 0;
        TN_DTree_ionosphere = 0;
        FN_DTree_ionosphere = 0;
        FP_DTree_ionosphere = 0; 
        for i = 1:length(Class_Test_ionosphere)
            if (Predicted_DTree_ionosphere(i,1) == 1 && Class_Test_ionosphere(i,1) == 1)
                TP_DTree_ionosphere = TP_DTree_ionosphere+1;
            end
            if (Predicted_DTree_ionosphere(i,1) == 2 && Class_Test_ionosphere(i,1) == 2)
                TN_DTree_ionosphere = TN_DTree_ionosphere+1;
            end
            if (Predicted_DTree_ionosphere(i,1) == 2 && Class_Test_ionosphere(i,1) == 1)
                FN_DTree_ionosphere = FN_DTree_ionosphere+1;
            end
            if (Predicted_DTree_ionosphere(i,1) == 1 && Class_Test_ionosphere(i,1) == 2)
                FP_DTree_ionosphere = FP_DTree_ionosphere+1;
            end
        end
        
%         Precision, Recall Calculation
        Precison_SVM_ionosphere(1,j) =  (TP_SVM_ionosphere)/(TP_SVM_ionosphere+FP_SVM_ionosphere);
        Recall_SVM_ionosphere(1,j) = (TP_SVM_ionosphere)/(TP_SVM_ionosphere+FN_SVM_ionosphere);
        Precison_kNN_ionosphere(1,j) =  (TP_kNN_ionosphere)/(TP_kNN_ionosphere+FP_kNN_ionosphere);
        Recall_kNN_ionosphere(1,j) = (TP_kNN_ionosphere)/(TP_kNN_ionosphere+FN_kNN_ionosphere);
        Precison_DTree_ionosphere(1,j) =  (TP_DTree_ionosphere)/(TP_DTree_ionosphere+FP_DTree_ionosphere);
        Recall_DTree_ionosphere(1,j) = (TP_DTree_ionosphere)/(TP_DTree_ionosphere+FN_DTree_ionosphere);
    end
    Mean_Precison_SVM_ionosphere(1,k-4) = mean(Precison_SVM_ionosphere);
    Mean_Recall_SVM_ionosphere(1,k-4) = mean(Recall_SVM_ionosphere);
    Mean_Precison_kNN_ionosphere(1,k-4) = mean(Precison_kNN_ionosphere);
    Mean_Recall_kNN_ionosphere(1,k-4) = mean(Recall_kNN_ionosphere);
    Mean_Precison_DTree_ionosphere(1,k-4) = mean(Precison_DTree_ionosphere);
    Mean_Recall_DTree_ionosphere(1,k-4) = mean(Recall_DTree_ionosphere);
end

% plot For Precision and Recall
k = 5:10;    
figure;plot(k,Mean_Precison_SVM_ionosphere);
title('precision Graph For ionosphere SVM');xlabel('k Value');ylabel('precision Value');
figure;plot(k,Mean_Recall_SVM_ionosphere);
title('Recall Graph For ionosphere SVM');xlabel('k Value');ylabel('Recall Value');
figure;plot(k,Mean_Precison_kNN_ionosphere);
title('precision Graph For ionosphere kNN');xlabel('k Value');ylabel('precision Value');
figure;plot(k,Mean_Recall_kNN_ionosphere);
title('Recall Graph For ionosphere kNN');xlabel('k Value');ylabel('Recall Value');
figure;plot(k,Mean_Precison_DTree_ionosphere);
title('precision Graph For ionosphere Decision Tree');xlabel('k Value');ylabel('precision Value');
figure;plot(k,Mean_Recall_DTree_ionosphere);
title('Recall Graph For ionosphere Decision Tree');xlabel('k Value');ylabel('Recall Value');