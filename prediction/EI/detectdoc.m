close all
clear all

clearvars record_RMpt2;
clearvars hdr_RMpt2;

%SUMMING PREDICTION RESULTS OF EIGHT RANDOMLY SELECTED CHANNELS
%predict seizure if half of channels predict seizure (output 1 at
%a particular time t)
%same channels and same parameters across the tests

%thresholding at 6 of 8 channels
%RMPT2 TESTS


%TEST 1
%load('TS041_03oct2010_05_34_02_Seizure.mat');
%seizure at 191s
%detection at 204s
%DETECTS 13 SECONDS AFTER

%TEST 2
%load('RMPt2_03oct2010_17_23_04_Seizure.mat');
%seizure at 107s
%detection at 119s
%DETECTS 12 SECONDS AFTER

%TEST 3
%load('RMPt2_03oct2010_01_03_05_Awake.mat');
%no seizure
%no detection - correct

%TEST 4
%load('RMPt2_04oct2010_11_12_02_Seizure.mat');
%seizure at 263s
%detection at 276s
%DETECTS 13 SECONDS AFTER

%TEST 5
%load('RMPt2_04oct2010_01_40_01_Sleep.mat');
%no seizure
%3 FALSE POSITIVES


%RMPT3 TESTS

%TEST 1
%load('TA533_30may2010_23_49_31_Seizure.mat');
%seizure at 360s
%detection at 350s
%PREDICTS 10 SECONDS BEFORE
%FALSE POSITIVE AT 43s


%TEST 2
%load('TA533_31may2010_09_25_18_Seizure.mat');
%seizure at 298s
%detection at 292s
%PREDICTS 6s SECONDS BEFORE

%for different patients:
    % ideal bias, threshold, and number of channels predicting 1 to
    % output a seizure prediction
v = .08;
L = .15;
time = 5;

%record_RMpt2 = record_RMpt3;


CH24_sz = record_RMpt2(24,:);
predict24 = epin(CH24_sz,v,L);

CH88_sz = record_RMpt2(88,:);
predict88 = epin(CH88_sz,v,L);

CH113_sz = record_RMpt2(113,:);
predict113 = epin(CH113_sz,v,L);

CH27_sz = record_RMpt2(27,:);
predict27 = epin(CH27_sz,v,L);

CH20_sz = record_RMpt2(20,:);
predict20 = epin(CH20_sz,v,L);

CH99_sz = record_RMpt2(99,:);
predict99 = epin(CH99_sz,v,L);

CH7_sz = record_RMpt2(7,:);
predict7 = epin(CH7_sz,v,L);

CH73_sz = record_RMpt2(73,:);
predict73 = epin(CH73_sz,v,L);

plot([zeros(1,time) predict73+predict24+predict88+predict113+predict27+predict20+predict7+predict99])
