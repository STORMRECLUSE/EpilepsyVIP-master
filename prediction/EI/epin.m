function [ predict ] = epin( CH1_sz, v, L)
%epin function:
%   inputs:
% CH1_sz = one channel of seizure data
% L = threshold value lambda
% v = bias value v
%   output:
% predict = vector of ones (seizure predicted)
%   and zeros (no seizure)

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


%% Step 1: Input Data

clearvars record_RMpt2;
clearvars hdr_RMpt2;
%working with one channel

%% Step 2: Sliding window

time = 2;   % 30 second window

W = time*1000;

c = 0;

ER_N = [0];
predict = [0 0 0];
for i=W+1:1000:length(CH1_sz)
    %makes sure each window size is consistent
    %takes window at each second
    c = c+1;
    
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
    ER_W(c) = (Beta+Gamma)/(Theta + Alpha);
    
    %% Step 7: Analyze when ER_N changes: (binary classification, CUSUM, etc.)
    %sum of ER statistic to this point
    ER_N = sum(ER_W)/length(ER_W);  
    %current value of ER minus running sum minus bias
    U_N(c) = ER_W(c)- ER_N - v;
    
    U_NINV = 1.01*max(U_N) - U_N;
    if length(U_NINV) > 2
        [pks, locs] = findpeaks(U_NINV);
        locs(1) = 1;
            if U_N(c) - U_N(locs(length(locs))) > L
        predict(c) = 1;
            else
                predict(c) = 0;
            end
    else
        predict(c) = 0;
    end
    
    
end

predict = [zeros(1,time) predict];

end
