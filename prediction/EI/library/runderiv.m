function RunDeriv=runderiv(Data,WindowWidth,TimeVec);
%-------------------------------------------------------------------------
% runderiv function       Calculate the runing derivative of an unevenly
%                       spaced time series, with flat weighting function
%                       (e.g., the slope in each window).
%                       Take into account slope-error and \chi^2.
% Input  : - Data matrix, [Time, Value, Error].
%            Error is optional, if not given then assumes
%            equal weights.
%          - Window total width in time units.
%          - Time vector for which to calculate the running mean,
%            default is Data(:,1).
% Output : - Runing derivative matrix:
%            [Time, Slope, Slope_err, N_points, Chi2DoF, Chi2CDF].
% Example: runderiv(Data);
% Tested : Matlab 5.3
%     By : Eran O. Ofek             January 2002
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%------------------------------------------------------------------------
TCol = 1;
YCol = 2;
ECol = 3;
PolyDeg = 1;
if (nargin==2),
   TimeVec = Data(:,TCol);
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end

DataSpan = max(Data(:,1)) - min(Data(:,1));
SizeData = size(Data);
if (SizeData(2)==2),
   Data = [Data, ones(SizeData(1),1)];
end


N = length(TimeVec);

RD  = zeros(N,1);
RDE = zeros(N,1);
RDN = zeros(N,1);
RDC = zeros(N,1);
RDP = zeros(N,1);
for I=1:1:N,
   J = find(abs(Data(:,TCol)-TimeVec(I))<=0.5.*WindowWidth);
   if (length(J)<2),
      RD(I)    = NaN;
      RDE(I)   = NaN;
      RDN(I)   = length(J);
      RDC(I)   = NaN;
   else
      MData = mean(Data(J,YCol));
      MTime = mean(Data(J,TCol));
      [P,PE,Cov,Chi2,DoF] = fitpoly(Data(J,TCol),Data(J,YCol),Data(J,ECol),PolyDeg);
      RD(I)    = P(2);               % slope
      RDE(I)   = PE(2);              % slope error
      RDN(I)   = length(J);          % number of points in window
      RDC(I)   = Chi2./DoF;          % Chi2 per DoF
      RDP(I)   = chi2cdf(Chi2,DoF);  % chi2 probability 
   end      
end

RunDeriv = [TimeVec, RD, RDE, RDN, RDC, RDP];
