% Algorithm based on Chisci et al. (2010)
% http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5415597&tag=1
% Version 4 - Seizure Detection
% Regularized SVM classification model on autoregression coefficients
% Copyright(c) 2015, Megan Kehoe
% Rice University VIP Team - mek2@rice.edu


% Step 1: Load data
% Step 2: Estimate autoregression (AR) coefficients every N samples
% Step 3: Calculate mean average (MA) of AR coefficients
% Step 4: Perform support vector machine (SVM) classification
% Step 5: Apply SVM model
% Step 6: Create white noise acceleration (WNA) state-space model
% Step 7: Apply Kalman filter (KF)
% Step 8: Calculate detection delay
% Step 9: Plot data


% **Change log for v4 implementation:

% Edited seizure labels from v3.2 to reflect neurologist's labels
% Performed cross validation to select parameters from C = [0.1 1 10],
% gamma = [0.0002 0.02 2], and m = [40 60 80] for patient TS041


% **Results:

% Cross validation chose C = 10, gamma = 2, and m = 40
% The seizure in the test set was detected by both standard SVM and KF
% regularized SVM
% Standard SVM detected seizure 13s before onset and had 1 false positive 
% in the inter-ictal period
% With KF regularization, detected seizure 71s after onset and had no false
% positives in the inter-ictal period
% Sensitivity on the test set was 77.8% (if defining each point in time as
% a training example) for standard SVM
% Specificity on the test set was 99.4% (if defining each point in time as
% a training example) for standard SVM
% Accuracy on the test set was 98.0% (if defining each point in time as
% a training example) for standard SVM


% **Notes on v4 implementation:

% Cross validation improved the algorithm's performance
% Used F1 score as error measurement to select parameters
% Intended to next test pre-ictal/ictal vs inter-ictal classification and
% perform cross validation on KF parameters to try to improve the detection
% time of KF regularized SVM


% **Next Steps:

% Test classification of (pre-ictal/ictal) vs (inter-ictal)
% Perform cross vaidation on a wider selection of values of parameters
% Perform cross validation of KF regularized classifier
% Include cross validation of parameters order and sigma_w
% Figure out if covariances are correct


function arSVM4

% ============ Step 1: Load Data ============ %

close all;
clear all;

% load seizure 11:12 TS041
clearvars record_RMpt2;
clearvars hdr_RMpt2;
load('TS041_04oct2010_11_12_02_Seizure.mat');
CH1_sz_11 = record_RMpt2(1,:); % take just first channel for now

% load seizure 17:23 TS041
clearvars record_RMpt2;
clearvars hdr_RMpt2;
load('TS041_03oct2010_17_23_04_Seizure.mat');
CH1_sz_17 = record_RMpt2(1,:);

% load seizure 5:34 TS041
clearvars record_RMpt2;
clearvars hdr_RMpt2;
load('TS041_03oct2010_05_34_02_Seizure.mat');
CH1_sz_5 = record_RMpt2(1,:); % take just first channel for now

% load inter-ictal sleep 2:00 TS041
clearvars record_RMpt2;
clearvars hdr_RMpt2;
load('TS041_05oct2010_02_00_08_Sleep.mat');
CH1_sleep_2 = record_RMpt2(1,:); % take just first channel for now

% load inter-ictal sleep 1:40 TS041
clearvars record_RMpt2;
clearvars hdr_RMpt2;
load('TS041_04oct2010_01_40_01_Sleep.mat');
CH1_sleep_1 = record_RMpt2(1,:);

% load inter-ictal sleep 3:00 TS041
clearvars record_RMpt2;
clearvars hdr_RMpt2;
load('TS041_02oct2010_03_00_05_Sleep.mat');
CH1_sleep_3 = record_RMpt2(1,:);

% load inter-ictal awake 9:00 TS041
clearvars record_RMpt2;
clearvars hdr_RMpt2;
load('TS041_02oct2010_09_00_38_Awake.mat');
CH1_awake_9 = record_RMpt2(1,:); % take just first channel for now

% load inter-ictal awake 1:03 TS041
clearvars record_RMpt2;
clearvars hdr_RMpt2;
load('TS041_03oct2010_01_03_05_Awake.mat');
CH1_awake_1 = record_RMpt2(1,:); % take just first channel for now

% load inter-ictal awake 13:05 TS041
clearvars record_RMpt2;
clearvars hdr_RMpt2;
load('TS041_03oct2010_13_05_13_Awake.mat');
CH1_awake_13 = record_RMpt2(1,:); % take just first channel for now


% select/combine training data
CH1_train = [CH1_sz_5 CH1_sleep_2 CH1_awake_1];

% select/combine cross validation data
CH1_cv = [CH1_sz_11 CH1_sleep_3 CH1_awake_13];

% select/combine test data
CH1_test = [CH1_sz_17 CH1_sleep_1 CH1_awake_9];


% ===================================================== %
% ============= PERFORM CROSS VALIDATION ===============%

accuracy = zeros(27,4);
F = zeros(27,1);
sen = zeros(27,1);
spec = zeros(27,1);
iter = 1;
for C = [0.1 1 10]
    for m = [40 60 80]
        for gamma = [0.0002 0.02 2]
            
            % ==== Steps 2 & 3: Preprocessing and Feature Extraction ===== %
            
            % training data
            [t_train, t_CH_train, AR_train] = calc(CH1_train);
            
            % cross validation data
            [t_cv, ~, AR_cv] = calc(CH1_cv);
            
            
            % ========== Step 4: Train SVM ============== %
            
            % create y vectors based on neurologist's marks in iEEG data
            % serve as labels for training data during SVM training
            if iter == 1
                % patient TS041 training data labels
                y_train = zeros(length(t_train),1);
                for i = 1:length(t_train)
                    if t_train(i) < 3.2 || t_train(i) > 6.75
                        y_train(i) = -1;
                    else
                        y_train(i) = 1;
                    end
                end
                % patient TS041 cross validation data labels
                y_cv = zeros(length(t_cv),1);
                for i = 1:length(t_cv)
                    if t_cv(i) < (4+23/60) || t_cv(i) > (5.5)
                        y_cv(i) = -1;
                    else
                        y_cv(i) = 1;
                    end
                end
            end
            
            % use AR coeffiecients of training set as features
            X = AR_train;
            
            % set parameters
            % -s 0 : C-SVC (C-paramaterized SVM)
            % -t 2 : Gaussian kernel
            % -g : value of gamma parameter
            % -c : value of C parameter
            param = ['-s 0 -t 2 -g ', num2str(gamma), ' -c ', num2str(C)];
            
            % train the model
            model = svmtrain(y_train, X, param);
            
            
            % ======== Step 5: Apply SVM Model ============ %
            
            % apply SVM model to cross validation data AR coefficients
            % y_cv included only for calculation of accuracy
            [pred, acc, ~] = svmpredict(y_cv, AR_cv, model);
            
            % record accuracy for each combination
            accuracy(iter,:) = [acc(1) C m gamma];
            
            % create array of structures to hold models of each combo
            if iter == 0
                model_hold = model;
            else
                model_hold(iter) = model;
            end
            
            % calculate F1 score for each combo
            % used F score because we have skewed data with higher number
            % of negative cases than positive cases
            % Need low false positives and low false negatives to have a
            % high F1 score
            [F(iter), sen(iter), spec(iter)] = fscore(pred,y_train);
            
            iter = iter + 1;
        end
    end
end

% select parameters with highest F1 score
[F,I] = max(F);
acc = accuracy(I,1);
C = accuracy(I,2);
m = accuracy(I,3);
gamma = accuracy(I,4);
model = model_hold(I);
sen = sen(I);
spec = spec(I);

% report results of cross validation
fprintf('Cross Validation Complete!\n')
fprintf('------\nF1 Score on Cross Validation Set = %f',F)
fprintf('\nAccuracy on Cross Validation Set = %f%',acc)
fprintf('\nSensitivity on Cross Validation Set = %f%',sen)
fprintf('\nSpecificity on Cross Validation Set = %f%',spec)
fprintf('\nOptimized Parameters:\n')
fprintf('C = %f | m = %f | gamma = %f\n------\n',C,m,gamma)

% ===================================================== %
% ===================================================== %



% ===================================================== %
% =============== PREDICTIONS ON TEST DATA =============%

% ==== Steps 2 & 3: Preprocessing and Feature Extraction ===== %

[t_test, t_CH_test, AR_test] = calc(CH1_test);


% ======== Step 5: Apply SVM Model ============ %

% patient TS041 test data labels
y_test = zeros(length(t_test),1);
for i = 1:length(t_test)
    if t_test(i) < (1+47/60) || t_test(i) > (4+47/60)
        y_test(i) = -1;
    else
        y_test(i) = 1;
    end
end

% apply SVM model to test data AR coefficients
% y_test included for calculation of accuracy
[pred, acc, ~] = svmpredict(y_test, AR_test, model);

% Report results
[F, sen, spec] = fscore(pred,y_test);
fprintf('SVM Predict Complete!\n')
fprintf('------\nF1 Score on Test Set = %f',F)
fprintf('\nAccuracy on Test Set = %f',acc(1))
fprintf('\nSensitivity on Test Set = %f%',sen)
fprintf('\nSpecificity on Test Set = %f%\n------\n\n',spec)

% ===================================================== %
% ===================================================== %



% =========== Steps 6 & 7: KF Regularization ===========%

% implement KF regularization
kf_pred = wna(5*10^-5,pred,t_test);


% ========== Step 8: Calculate detection delay ============= %

% without KF regularization
fprintf('\nWithout KF regularization:\n')
y2 = detect_time(t_CH_test,t_test,pred);

% with KF regularization
fprintf('\nWith KF regularization:\n')
y2 = detect_time(t_CH_test,t_test,kf_pred);


% =========== Step 9: Plot data ============== %

% plot training data
figure
hold on
plot(t_CH_train,CH1_train/5000,'Color',[0.8 0.8 0.8]) % iEEG data
plot(t_train, AR_train) % AR coefficients
plot(t_train, y_train, 'Color', [0 0 0], 'LineWidth', 2) % Data labels
title('Training Data')
xlabel('Time (min)')
legend('iEEG','AR_0','AR_1','AR_2','AR_3','AR_4','AR_5','AR_6','y',...
    'Location','eastoutside')
ylim([-2 2])
hold off

% plot cross validation data
[t_cv, t_CH_cv, AR_cv] = calc(CH1_cv);
[pred_cv, ~, ~] = svmpredict(y_cv, AR_cv, model);
figure
hold on
plot(t_CH_cv,CH1_cv/5000,'Color',[0.8 0.8 0.8]) % iEEG data
plot(t_cv, AR_cv) % AR coefficients
plot(t_cv, y_cv, 'Color', [1 0.3 0.7], 'LineWidth', 2) % Data labels
plot(t_cv, pred_cv, 'Color', [0 0 0], 'LineWidth', 2) % Predicition
title('Cross Validation Data')
xlabel('Time (min)')
legend('iEEG','AR_0','AR_1','AR_2','AR_3','AR_4','AR_5','AR_6','y',...
    'Location','eastoutside')
ylim([-2 2])
hold off

% plot the results of standard SVM
figure
hold on
plot(t_CH_test,CH1_test/5000,'Color',[0.8 0.8 0.8]) % iEEG data
plot(t_test,AR_test) % AR coefficients
plot(t_CH_test, y2, 'Color', [1 0.3 0.7], 'LineWidth', 2) % Data labels
plot(t_test, pred, 'Color', [0 0 0], 'LineWidth', 2) % Predicition
title('Test Data - Standard SVM')
xlabel('Time (min)')
legend('iEEG','AR_0','AR_1','AR_2','AR_3','AR_4','AR_5','AR_6','Labels',...
    'Prediction','Location','eastoutside')
ylim([-2 2])
hold off

% plot the results of KR regularization
figure
hold on
plot(t_CH_test,CH1_test/5000,'Color',[0.8 0.8 0.8]) % iEEG data
plot(t_test,AR_test) % AR coefficients
plot(t_CH_test, y2, 'Color', [1 0.3 0.7], 'LineWidth', 2) % Data labels
plot(t_test, kf_pred, 'Color', [0 0 0], 'LineWidth', 2) % Predicition
title('Test Data - KF Regularization')
xlabel('Time (min)')
legend('iEEG','AR_0','AR_1','AR_2','AR_3','AR_4','AR_5','AR_6','Labels',...
    'Prediction','Location','eastoutside')
ylim([-2 2])
hold off

end


% preprocessing and feature extraction
function [t_ma, t_original, AR] = calc(CH1)

% ======= Step 2: Estimate AR Coefficients ======== %

fs = 1000; % sampling frequency
N = 500; % calculate AR coefficients every N samples
order = 6; % order of AR model

% Calculate AR coefficients using aryule over every N samples
AR_pre = zeros(floor(length(CH1)/N),order + 1);
j = 1;
for i = 1:length(CH1)
    if rem(i,N) == 0
        CH1_temp = CH1(i-N+1:i);
        AR_temp = aryule(CH1_temp,order);
        AR_pre(j,:) = AR_temp;
        j = j + 1;
    end
end


% ========== Step 3: Calculate Mean Average ========== %

% Calculate MA over every m samples
m = 60;
j = 1;
AR = zeros(length(AR_pre)-m+1,order + 1);
for i = m:size(AR_pre,1)
    AR_temp2 = AR_pre(i-m+1:i,:);
    AR_temp2 = sum(AR_temp2)/m;
    AR(j,:) = AR_temp2;
    j = j + 1;
end

% mean centering of features
AR_mean = sum(AR)/size(AR,1);
for i = 1:length(AR_mean)
    AR(:,i) = AR(:,i) - AR_mean(i);
end

% time vectors in minutes
t_ma = ((0:length(AR)-1) /fs * N)/60; % for use with AR
t_original = ((0:(length(CH1)-1)) / fs)/60;% for use with raw iEEG data
end


% Kalman filtering on WNA state-space model
function kf_pred = wna(sigma_w,y,t)

% ========= Step 6: Create WNA State-Space Model ==========%

% parameters of discrete WNA model
Tp = 0.5/60; % sampling time = 0.5s
A = [1 Tp; 0 1];
B = [0.5*Tp^2; Tp];
C = [1 0];
D = 1;

% create state-space model
sys = ss(A,B,C,D,Tp);

% =========== Step 7: Apply Kalman Filter ============= %

% covariance matricies of WNA model
% Q = sigma_w^2*[Tp^3/3 Tp^2/2; Tp^2/2 Tp];
sigma_v = 1;
R = sqrt(sigma_v);

% create the kalman state-space model kest
[kest,~,~] = kalman(sys,sqrt(sigma_w),R);
kest = kest(1,:);

% obtain the filtered estimate dk of the decision variable
dk = lsim(kest,y,t);

% sign is a decision function
% if dk > 0, kf_pred = 1
% otherwise, kf_pred = -1
kf_pred = sign(dk);

end


% calculate seizure detection time
function y2 = detect_time(t_CH_test,t_test,pred)

% create vector of labels based on neurologist's markings to compare
% detection result to

% patient TA023
y2_023 = zeros(1,length(t_test));
for i = 1:length(t_CH_test)
    if t_CH_test(i) < 6+37/60 || t_CH_test(i) > 7+7/60
        y2_023(i) = -1;
    else
        y2_023(i) = 1;
    end
end

% patient TS041
y2_041 = zeros(1,length(t_test));
for i = 1:length(t_CH_test)
    if t_CH_test(i) < 1+47/60 || t_CH_test(i) > 4+47/60
        y2_041(i) = -1;
    else
        y2_041(i) = 1;
    end
end

% select current patient
y2 = y2_041;

% calculate seizure onset time
onset = -1;
for i = 1:length(y2)
    while y2(i) == 1 && onset == -1;
        onset = t_CH_test(i);
        break
    end
end

% calculate seizure detection time
detect = -1;
for i = 1:length(pred)
    while pred(i) == 1 && detect == -1;
        detect = t_test(i);
        break
    end
end

% calculate difference in onset and detection times
delay = (detect - onset)*60;

% report detection result
if onset == -1
    fprintf('No seizure present in test file')
end

if detect == -1
    fprintf('No seizure detected\n')
else if delay >= 0
        fprintf('Detected seizure %f seconds after onset\n',delay)
    else
        fprintf('Detected seizure %f seconds before onset\n',-delay)
    end
end
end


% calculate F1 score
function [F, sen, spec] = fscore(pred,y_train)

% calculate true positives, true negatives, false positives,
% and false negatives
TP = 0;
TN = 0;
FP = 0;
FN = 0;
for i = 1:length(pred)
    if pred(i) == 1 && y_train(i) == 1
        TP = TP + 1;
    else if pred(i) == 1 && y_train(i) == -1
            FP = FP + 1;
        else if pred(i) == -1 && y_train(i) == -1
                TN = TN + 1;
            else if pred(i) == -1 && y_train(i) == 1
                    FN = FN + 1;
                end
            end
        end
    end
end

% calculate precision and recall
prec = TP/(TP+FP);
rec = TP/(TP+FN);

% calculate F1 score
F = 2*prec*rec/(prec+rec);

% calculate sensitivity and specificity
sen = TP/(TP+FN)*100;
spec = TN/(TN+FP)*100;
end
