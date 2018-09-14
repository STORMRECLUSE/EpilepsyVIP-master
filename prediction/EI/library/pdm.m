function PDM=pdm(Data,FreqVec,BinNum);
%--------------------------------------------------------------------
% pdm function     phase dispersion minimization
%                Period searching by folding the time series into
%                a trail periods and calculating the dispersion of
%                the light curve.
% Input  : - Data matrix, first column for time and second for magnitude.
%          - Frequency vector, [minimum_freq, maximum_freq, delta_freq]
%	   - Number of bins in the folded period.
% Output : - Matrix in which the first column is frequency,
%            the second is the dispersion indicator normalized
%            by the variance of the data.
%            for a true period theta=1.
% Reference: Stellingwerf, R.F. ApJ 224, 953-960 (1978).
%            Schwarzenberg-czerny, A. ApJL, 489, 941-945 (1997)
% see also : minds function.
% Tested   : Matlab 5.0
%       By : Eran O. Ofek           June 1994
%      URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
Var = std(Data(:,2),0).^2;
Ntot = length(Data(:,1));
D0 = Ntot - 1;
D1 = BinNum - 1;
D2 = Ntot - BinNum;

PDM = zeros(size([FreqVec(1):FreqVec(3):FreqVec(2)],2));

I = 0;
for Freq=FreqVec(1):FreqVec(3):FreqVec(2),
   I = I + 1;
   Folded = [Data(:,1).*Freq - fix(Data(:,1).*Freq), Data(:,2)];
   [BFData] = binning(Folded,1./BinNum,0,1);

   N   = BFData(:,7);
   StD = BFData(:,3).*sqrt(BFData(:,7));
   S2  = sum((N-1).*StD.^2)./(sum(N) - BinNum);
   Theta = S2./Var;
   PDM(I,1:2) = [Freq, Theta];
end
