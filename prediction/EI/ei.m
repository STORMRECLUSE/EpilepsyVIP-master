% Epileptogenicity Index Algorithm
% http://brain.oxfordjournals.org/content/brain/131/7/1818.full.pdf

% Step 1: Input data
% Step 2: Sliding window
% Step 3: FFT: get X(w)
% Step 4: Multiply: X(w)X*(w) / 2pi gets ESD
% Step 5: Integrate over bands over frequency interval (look at paper for numbers)
%             Four times: gamma beta theta alpha
% Step 6: Create ratio ER
% Step 7: Analyze when ER changes cumulative sum function or other matlab standard
% Step 8: Outputs prediction of seizure


close all
clear all

%% Step 1: Input Data

clearvars record_RMpt2;
clearvars hdr_RMpt2;
%working with one channel

%TEST 1
% load('TS041_03oct2010_05_34_02_Seizure.mat');
% CH1_sz = record_RMpt2(1,:);
% % edf file: DA00101W_1-1+
% % seizure at 3:11 = 191 seconds
% % earliest detection at 101 seconds
% % also detection at 168 seconds

%TEST 2
% load('RMPt2_03oct2010_17_23_04_Seizure.mat');
% CH1_sz = record_RMpt2(55,:);
% edf file: DA00101V_1-1+
% seizure at 1:47 = 107 seconds
% earliest detection at 83 seconds

%TEST 3
% load('RMPt2_03oct2010_01_03_05_Awake.mat');
% CH1_sz = record_RMpt2(1,:);
% %no seizure
% %a couple false alarms, can change threshold

%TEST 4
load('RMPt2_04oct2010_11_12_02_Seizure.mat');
CH1_sz = record_RMpt2(64,:);
% %edf file: DA00101U_1-1+
% %seizure at 4:23 = 263 seconds
% %earliest detection at 243 seconds

%looking at jumps greater than ~.07, around ~.25
%TEST 4 seems to require different parameters than TESTS 1-3
%different threshold greatly improves accuracy on TESTS 1-3
%but completely misses the TEST 4 seizure
%v = .08 L = .18 for TESTS 1-3
%v = .1 L = .08 for all TESTS

%% Step 2: Sliding window

time = 5;   % 30 second window

W = time*1000;

ER_N = [0];
detection = [0 0 0];
for i=W+1:1000:length(CH1_sz)  
    %makes sure each window size is consistent
    %takes window at each second
    
    Window = CH1_sz(i-W:i);    
    %% Step 3: Take FFT: get X(w) 
    X_w = fft(Window);
    
    %% Step 4: Get ESD - Energy Spectral Density
    ESD = (X_w.*conj(X_w))/(2*pi);
    
    %% Step 5: Integrate over bands over frequency interval
        %Four times: gamma beta theta alpha
    %Section out elements of ESD corresponding to each frequency band
    %index * (fs/2) * (1/W) = frequency
    ESDt = ESD(round((W/500)*3.5):(round((W/500)*7.4)));
    ESDa = ESD(round((W/500)*7.4):(round((W/500)*12.4)));
    ESDb = ESD(round((W/500)*12.4):(round((W/500)*24)));
    ESDg = ESD(round((W/500)*24):(round((W/500)*97)));
    %Do time-series integration using trapz function
        %Calculates total sum of area between each data point, each
        %of which look like a trapezoid
    Theta = trapz(ESDt);
    Alpha = trapz(ESDa);
    Beta =  trapz(ESDb); 
    Gamma = trapz(ESDg);
    
    %% Step 6: Create energy ratio ER
    ER_W(((i-1)/1000)+1) = (Beta+Gamma)/(Theta + Alpha);
    
    %% Step 7: Analyze when ER_N changes: (binary classification, CUSUM, etc.)
    %sum of ER statistic to this point
    ER_N = sum(ER_W)/length(ER_W);
    %set bias v
    v = .08;
    %set threshold lambda
    L = .18;     
    %current value of ER minus running sum minus bias
    U_N(((i-1)/1000)+1) = ER_W(((i-1)/1000)+1)- ER_N - v;
    
    
%     if U_N(((i-1)/1000)+2) - U_N(((i-1)/1000)+1) > L
%         detection(((i-1)/1000)+1) = 1;
%     else
%         detection(((i-1)/1000)+1) = 0;
%     end
    
    %U_N(((i-1)/1000)+1) = sum(ER_W) - ER_N - v;
    
    
    U_NINV = 1.01*max(U_N) - U_N;
    if length(U_NINV) > 2
        [pks, locs] = findpeaks(U_NINV);
        locs(1) = 1;
            if U_N(((i-1)/1000)+1) - U_N(locs(length(locs))) > L
        detection(((i-1)/1000)+1) = 1;
            else
                detection(((i-1)/1000)+1) = 0;
            end
    else
        detection(((i-1)/1000)+1) = 0;
    end
    
    
end

%% Step 8: Outputs prediction of seizure
figure
plot(1:(length(CH1_sz)+ W)/1000,[zeros(1,time) ER_W])
title('Energy Ratio statistic over time')
xlabel('Time (s)')
ylabel('High Frequency Energy over Low Frequency Energy')
figure(2)
plot(1:(length(detection)+time),[zeros(1,time) detection])
title('Seizure Prediction')
xlabel('Time (s)')
ylabel('<- No Seizure                     Seizure Predicted ->')


