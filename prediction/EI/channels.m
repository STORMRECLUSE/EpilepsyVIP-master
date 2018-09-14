close all
clear all

clearvars record_RMpt2;
clearvars hdr_RMpt2;

%SUMMING PREDICTION RESULTS OF EIGHT RANDOMLY SELECTED CHANNELS
%predict seizure if half of channels predict seizure (output 1 at
%a particular time t)
%same channels and same parameters across the tests

%TEST 1
%load('TS041_03oct2010_05_34_02_Seizure.mat');
% L = .15 or .18
%8/8 channels predict at 170-172 seconds - true positive
%oscillations are up to two channels predicting at a time

%TEST 2
load('RMPt2_03oct2010_17_23_04_Seizure.mat');
% L = .15
%7/8 channels predict at 85 seconds - true positive
%6/8 channels predict at 286 seconds - false positive
%other oscillations are only up to one channel predicting at a time

% L = .18
%6/8 channels predict at 84-85 seconds - true positive
%oscillations are up to only one channel predicting at a time

%TEST 3
%load('RMPt2_03oct2010_01_03_05_Awake.mat');
% L = .15 or .18
%no more than one channel predicts at a time

%TEST 4
%load('RMPt2_04oct2010_11_12_02_Seizure.mat');
% L = .15
%7/8 channels predict at 243 seconds - true positive
%oscillations are up to only one channel predicting at a time

% L = .18
%5/8 channels predict at 243 seconds - true positive
%oscillations are up to only one channel predicting at a time

%TEST 5
%load('RMPt2_04oct2010_01_40_01_Sleep.mat');
% L = .15 or .18
%no more than one channel predicts at a time

%for different patients:
    % ideal bias, threshold, and number of channels predicting 1 to
    % output a seizure prediction
v = .08;
L = .15;
time = 2;

CH5_sz = record_RMpt2(24,:);
predict5 = epin(CH5_sz,v,L);

CH65_sz = record_RMpt2(88,:);
predict65 = epin(CH65_sz,v,L);

CH110_sz = record_RMpt2(113,:);
predict110 = epin(CH110_sz,v,L);

CH26_sz = record_RMpt2(27,:);
predict26 = epin(CH26_sz,v,L);

CH89_sz = record_RMpt2(20,:);
predict89 = epin(CH65_sz,v,L);

CH100_sz = record_RMpt2(99,:);
predict100 = epin(CH100_sz,v,L);

CH42_sz = record_RMpt2(7,:);
predict42 = epin(CH42_sz,v,L);

CH1_sz = record_RMpt2(73,:);
predict1 = epin(CH1_sz,v,L);

plot([zeros(1,time) predict1+predict5+predict65+predict42+predict100+predict110+predict26+predict89])
