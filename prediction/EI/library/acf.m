afoevfor.m                                                                                          0100644 0056337 0000144 00000001641 07703072372 011734  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function y=afoevfor2(file);
%--------------------------------------------------------------------
% afoevfor function    formating AFOEV observations into
%                    matlab variable.
% Input  : - file name, contains the variable star observation
%            in the AFOEV format.
% Output : - matrix of observations, in which the first column
%            contains the MJD and the second column contains the
%            visual magnitude.
% Tested : Matlab 5.0
%     By : Eran O. Ofek           July 1995
%  modified by Orly Gnat        August 1997
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
fid = fopen(file,'r');
line = 0;
while line~=-1,
   line = fgetl(fid);
   if line==-1,
      break;
   end
   if line(12:13)==' ',
      jd = str2num(line(4:11));
      mg = str2num(line(14:16));
      y  = [y;[jd,mg]];
   end
end
fclose(fid);

                                                                                               arp.m                                                                                               0100644 0056337 0000144 00000003424 07703073773 010716  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [a,cer,th]=arp(x,p,ex);
%--------------------------------------------------------------------
% arp function     autoregessive process of order p.
%                moddeling an evenly spaced time series, with:
%                z(t) = sigma[a(i)*z(t-i)] + e(t) 
% input  : - Matrix in which the first column is index (time)
%            and the second column is the observation value.
%	   - The autoregressive order (lag to run on). default is N/4.
%	   - The number of points beyond the last point of the series
%            to extraplate the series. Default is zero.
% output : - AR(p) parameters.
%	   - AR(p) error in the parameters.
%	   - Series extrapolation into the future.
%	     The first column is index and the second is for the
%            predicted value of the series. 
% reference : Koen, C. & Lombard, F. 1993 MNRAS 263, 287-308
% Tested : Matlab 4.2
%     By : Eran O. Ofek           September 1994
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
c_x = 1;
c_y = 2;
N = length(x(:,c_x));
if nargin==1,
   p   = floor(N/4);
elseif nargin==2,
   c_x = 1;
elseif nargin>3,
   error('1, 2 or 4 args only');
end
% x = sortby(x,c_x);
y = x(:,c_y);
if p>N/4,
   'The lag is exceding size/4 of series'
end
my = mean(y);
zt = y - my;
z = zeros(N-p,p);
k = 1;
for i=p+1:N,
   z(k,:) = rot90(zt(i-p:i-1),3);
   k = k + 1;
end
zts = zt(p+1:N,1);
a=z\zts;
cer = sqrt(diag(cov(z)));
if (nargin==3 & nargout==3),
   sp = x(2,c_x) - x(1,c_x);
   st = x(N,c_x) + sp;
   th = zeros(ex,2);
   for j=1:ex,
      th(j,c_x) = st + sp.*(j - 1);
      th(j,c_y) = rot90(a,1)*rot90(zt(i+j-p:i+j-1),2);
      zt(N+j)   = th(j,c_y);
   end
elseif (nargin==3 & nargout~=3),
   error('theoretical extension needed two output args')
end
                                                                                                                                                                                                                                            bin_by_eye.m                                                                                        0100644 0056337 0000144 00000004004 07722702456 012231  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function B=bin_by_eye(Data);
%-------------------------------------------------------------------------
% bin_by_eye function       Plot data and define binning by eye.
%                         The user mark (with the mouse) the beginning
%                         and end points of each bin.
%                         The left and right limits of each bin defined
%                         by the user are marked by cyan and red
%                         dashed lines, respectively.
% Input  : - Data matrix, [Time, Value, Error].
% Output : - Matrix with the following columns:
%            [Bin_Center,
%             <Y>,
%             StD(Y)/sqrt(N),
%             <X>,
%             Weighted-<Y>,
%             Formal-Error<Y>,
%             N]
%            In case that no errors are given, columns 5 and 6
%            will be zeros.
% Tested : Matlab 5.3
%     By : Eran O. Ofek
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%-------------------------------------------------------------------------
ColT = 1;
ColM = 2;
ColE = 3;

errorxy(Data,[ColT, ColM, ColE],'o');
hold on;
YLim = get(gca,'YLim');

R = 'y';

zoom on;
disp(sprintf('\n Zoom is on, use maouse to set zoom'));
R = input('  Strike y to continue, other keys to stop\n','s');

B       = zeros(0,7);
LoopInd = 0;
while (R=='y'),
   LoopInd = LoopInd + 1;
   disp(sprintf('\n'));
   disp(sprintf('\n mark beginning and end points for current bin'));
   [X,Y] = ginput(2);
   if (X(1)>X(2)),
      T1 = X(2);
      T2 = X(1);
   else
      T2 = X(2);
      T1 = X(1);
   end
   I       = find(Data(:,ColT)>=T1 & Data(:,ColT)<T2);
   SubData = Data(I,:); 

   [WM,WE] = wmean(SubData(:,[ColM, ColE]));
   N       = length(SubData(:,1));
   B(LoopInd,:) = [0.5.*(T1+T2),...
   	           mean(SubData(:,ColM)),...
		    std(SubData(:,ColM))./sqrt(N),...
		   mean(SubData(:,ColT)),...
		   WM,...
		   WE,...
		   N];   

   plot([T1,T1],YLim,'c--');
   plot([T2,T2],YLim,'r--');


   % next bin: y/n
   R = input('  Strike y to continue, other keys to stop\n','s');
end


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            binning.m                                                                                           0100644 0056337 0000144 00000006211 10301131557 011534  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [ResMat,NewMat]=binning(Data,BinSize,Start,End);
%--------------------------------------------------------------------
% binning function     Binning a timeseries, with eqal size,
%                    equal weight bins.
%                    Calculate the mean/median/std/skewness
%                    of the observations in each bin.
%                    The bins are equally spaced,
% Input  : - Matrix, in which the first column is the
%            time, the second column is the observation, and
%            optional third column is the value error.
%          - binning interval, in units of the "time" column.
%          - First bin start Time (default is start point).
%          - last bin end Time (default is end point).
% Output : - Matrix with the following columns:
%            [Bin_Center,
%             <Y>,
%             StD(Y)/sqrt(N),
%             <X>,
%             Weighted-<Y>,
%             Formal-Error<Y>,
%             N]
%            In case that no errors are given, columns 5 and 6
%            will be zeros.
%          - The matrix as above, but the NaNs are eliminated.
% Tested : Matlab 5.0
%     By : Eran O. Ofek       Febuary 1999 / Last Updated Feb 2000
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------

Xcol = 1;
Ycol = 2;
Ecol = 3;

if nargin<2,
   error('at least 2 args');
end
if nargin==2,
   Start = min(Data(:,Xcol));
   End   = max(Data(:,Xcol));
end
if nargin==3,
   End   = max(Data(:,Xcol));
end

if (length(Data(1,:))==2),
   % without errors
   Err = 'n';
else
   Err = 'y';
end

Ntot = length(Data(:,Xcol));

Nbin = round((End-Start)./BinSize);

% Bin_Center; <X>; <Y>; <Y^2>; N
BinMat = zeros(Nbin,7);

% initialize bin center
BinMat(:,1) = [(Start+0.5.*BinSize):BinSize:(End-0.5.*BinSize)]';

ResMat = BinMat;


for I=1:1:Ntot,
   BinInd = ceil((Data(I,Xcol)-Start)./BinSize);
   if (BinInd>0 & BinInd<=Nbin),
      BinMat(BinInd,2) = BinMat(BinInd,2) + Data(I,Xcol);
      BinMat(BinInd,3) = BinMat(BinInd,3) + Data(I,Ycol);
      BinMat(BinInd,4) = BinMat(BinInd,4) + Data(I,Ycol).^2;
      BinMat(BinInd,5) = BinMat(BinInd,5) + 1;
      % for w-mean and formal error
      if (Err=='y'),
         BinMat(BinInd,6) = BinMat(BinInd,6) + Data(I,Ycol)./(Data(I,Ecol).^2);
         BinMat(BinInd,7) = BinMat(BinInd,7) + 1./(Data(I,Ecol).^2);
      end
      % for skewness
      %BinMat(BinInd,8) = BinMat(BinInd,4) + Data(I,Ycol).^3;
   end
end

% calculate mean and StD
% Bin_Center; <Y>; StD(Y)/sqrt(N); <X>; W-<Y>; Formal-Error(Y); N; Skewness(Y)
ResMat(:,2) = BinMat(:,3)./BinMat(:,5);
ResMat(:,3) = sqrt((BinMat(:,4)./BinMat(:,5) - ResMat(:,2).^2)./BinMat(:,5));
ResMat(:,4) = BinMat(:,2)./BinMat(:,5);
if (Err=='y'),
   % W-mean
   ResMat(:,5) = BinMat(:,6)./BinMat(:,7);
   % formal error
   ResMat(:,6) = sqrt(1./BinMat(:,7));
end
% Number of observations per bin
ResMat(:,7) = BinMat(:,5);
% The skewness
%ResMat(:,8) = (BinMat(:,8) - 3.*BinMat(:,4).*ResMat(:,2)-3.*BinMat(:,3).*ResMat(:,2).^2 - ResMat(:,2).^3)./(ResMat(:,7).*ResMat(:,3).^3);

if (nargout>1),
   % eliminate NaNs
   Innan  = find(isnan(ResMat(:,2))==0);
   NewMat = ResMat(Innan,:);
end

   


                                                                                                                                                                                                                                                                                                                                                                                       binn.m                                                                                              0100644 0056337 0000144 00000002450 07703076532 011054  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function y=binn(x,n);
%--------------------------------------------------------------------
% binning function     Binning a set of observations by equal number
%                    of observations within each bin.
%                    The program returns a matrix containing
%                    the mean "time", mean value and value standard
%                    deviation.
%                    If the number of observations is not divided by
%                    the number of points in each bin without
%                    a reminder, then the last remaining observations
%                    will not be used.
% Input  : - Matrix [Time, Value], sorted by the time column.
%          - Number of points in each bin.
% Output : - Binned matrix, [Time Value StD].
% Tested : Matlab 5.0
%     By : Eran O. Ofek           August 1996
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
if nargin~=2,
   error('2 args only');
end
c_x = 1;
c_y = 2;
nx = length(x(:,1));
% nb is total number of bins
nb = floor(nx./n);
% ntu is Number of observation To Use
ntu = n.*nb;
x = x(1:ntu,:);
y = zeros(nb-1,3);
for i=1:1:nb-1,
   y(i,:) = [mean(x(1+(i-1)*n:1+(i-1)*n+n,c_x)), mean(x(1+(i-1)*n:1+(i-1)*n+n,c_y)), std(x(1+(i-1)*n:1+(i-1)*n+n,c_y))];
end



                                                                                                                                                                                                                        ccf.m                                                                                               0100644 0056337 0000144 00000012005 10256555220 010647  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [CCFMat]=ccf(Ser1,Ser2,BinSize,Type,Correct,MaxDeltaT);
%--------------------------------------------------------------------
% ccf function    Discrete Cross-Correlation Function,
%                cross correlate two sets of stationary time series.
%                with unequal spacing and unequal series length.
%                Using the "binning-method". 
% input  : - first series matrix:
%            [Time, Mag, Err], the third column (Err) is optional,
%            if not given, assumes Err=0.
%          - second series matrix:
%            [Time, Mag, Err], the third column (Err) is optional,
%            if not given, assumes Err=0.
%          - BinSize is the delta lag on which the CCF is calculated.
%          - Type of correlation:
%            'normal' : normal correlation, error given by:
%                       (1-rho^2)/sqrt(N-1), (default).
%            'z'      : z-transform, rho is transformed to z
%                       (using z-transform), the errors are given
%                       by 1/sqrt(N-3) has more Gaussian shape.
%                       (Barlow 1989, p. 80)
%          - Correct structure function to measurments errors
%            {'y' | 'n'}, default is 'y'.
%          - Maximum lag to check, default is no limit (i.e., NaN).
% output : - CCF matrix:
%            [Mid. Lag, CCF, Err, point in bin-lag, Mean Lag, Struct_fun, err_struct_fun].
%            Note that: The structure function has units of amplitude^2.
%                       The structure function is corrected for the
%                       measurments errors, so it has zero amplitude
%                       at zero lag.
% Reference : Edelson, R.A. & Krolik, J.H. 1988 MNRAS 333, 646-659.
%             Koen, C. & Lombard, F. 1993 MNRAS 263, 287-308.
% See Also  : ccf_o.m (old version - for equally spaced...)
% Tested : Matlab 5.3
%     By : Eran O. Ofek           June 1999
%                    Last Update: May 2001
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
MinEl = 5;

if (nargin==3),
   Type      = 'normal';
   Correct   = 'y';
   MaxDeltaT = 'NaN';
elseif (nargin==4),
   Correct   = 'y';
   MaxDeltaT = 'NaN';
elseif (nargin==5),
   MaxDeltaT = 'NaN';
elseif (nargin==6),
   % do nothing
else
   error('Illigal number of input arguments');
end

if (length(Ser1(:,1))<MinEl | length(Ser2(:,1))<MinEl),
   % return empty - do not calculate ccf
   CCFMat = [];
else

   T1 = Ser1(:,1);
   T2 = Ser2(:,1);
   M1 = Ser1(:,2) - mean(Ser1(:,2));
   M2 = Ser2(:,2) - mean(Ser2(:,2));
   if (size(Ser1,2)>=3 & size(Ser2,2)>=3),
      % Take errors into account
      E1 = Ser1(:,3);
      E2 = Ser2(:,3);
   
      TakeError = 1;
   else
      E1 = 0;
      E2 = 0;
      TakeError = 0;
   end
   
   
   
   StD1  = std(M1);
   StD2  = std(M2);
   Norm  = sqrt((StD1.^2 - mean(E1).^2).*(StD2.^2 - mean(E2).^2));
   
   N1 = length(T1);
   N2 = length(T2);
   
   MaxTimeSpan = max(T2)-min(T1);
   
   TimeBoundry = [[-rot90([BinSize:BinSize:MaxTimeSpan].',2)]; [0:BinSize:MaxTimeSpan].'];
   TimeLag     = TimeBoundry(1:end-1) + diff(TimeBoundry);
   N_Cell  = length(TimeLag);
   
   M          = zeros(N_Cell,1);
   LagOffset  = zeros(N_Cell,1);
   DCF        = zeros(N_Cell,1);
   UDCF       = zeros(N_Cell,1);
   UDCF2      = zeros(N_Cell,1);
   %CCFMat     = zeros(N_Cell,4);
   StrFun     = zeros(N_Cell,1);
   StrFunErr2 = zeros(N_Cell,1);
   
   
   for I=1:1:N1,
      for J=1:1:N2,
         DeltaT = T2(J) - T1(I);
         if (abs(DeltaT)<=MaxDeltaT),
   
            [MinVal, TimeLagInd]   = min(abs(DeltaT - TimeLag));
            LagOffset(TimeLagInd)  = LagOffset(TimeLagInd) + MinVal;
            M(TimeLagInd)          = M(TimeLagInd) + 1;
            UDCF(TimeLagInd)       = UDCF(TimeLagInd) + M1(I).*M2(J)./Norm;
            UDCF2(TimeLagInd)      = UDCF2(TimeLagInd) + (M1(I).*M2(J)./Norm).^2;
	    StrFun(TimeLagInd)     = StrFun(TimeLagInd) + (M1(I) - M2(J)).^2;
	    StrFunErr2(TimeLagInd) = StrFunErr2(TimeLagInd) + (E1(I).*2.*(M1(I)-M2(J))).^2 + (E2(J).*2.*(M1(I)-M2(J))).^2;
         end                       
      end
   end
   
   LagOffset   = LagOffset./M;
   DCF         = UDCF./M;
   StrFun      = StrFun./M; 
   StrFunErr2  = sqrt(StrFunErr2./(M.^2)); 

   % correct structure function, so it will start with zero amplitude (at zero lag).
   
   switch Correct
    case 'y'
       StrFunShift = (mean(E1).^2 + mean(E2).^2);
       StrFun      = StrFun - StrFunShift;
    case 'n'
       % do nothing
    otherwise
       error('Unknown Correct option');
   end


   % Edelson & Krolik formula:
   %ErrDCF = sqrt(UDCF2 - 2.*UDCF.*DCF + DCF.^2)./(M-1);
   
   % correct for abs(rho)>1
   I = find(abs(DCF)>1);
   DCFe = DCF;
   DCFe(I) = 1;
   
   switch Type
    case 'normal'
       ErrDCF = (1-DCFe.^2)./sqrt(M-1);
    case 'z'
       % z-transform
       DCF    = 0.5.*log((1+DCF)./(1-DCF));
       ErrDCF = 1./sqrt(M-3);
    otherwise
       error('Unknown CCF type');
   end
   
   CCFMat = [TimeLag, DCF, ErrDCF, M, TimeLag+LagOffset, StrFun, StrFunErr2];
   
   
   
end
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ccf_o.m                                                                                             0100644 0056337 0000144 00000003201 07703077432 011172  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [rk,Q,S]=ccf_o(x1,x2,J);
%--------------------------------------------------------------------
% ccf_o function     Cross-Correlation function
%                  cross correlate two sets of equally
%                  spaced time series. 
% input  : - two column matrix in which the first column, is
%            an arbitrary index, and the second column is
%            the variable to correlate.
%          - second matrix of observations.
%	   - lag to run on (default is N/4).
% output : - Two column matrix, in which the first column contains
%            the lag and the second the correlation.
%	   - Pormanteau statistics.
%	   - The standard deviation of the ccf (for very large N).
% reference : Koen, C. & Lombard, F. 1993 MNRAS 263, 287-308
% See Also  : acf.m, ccf.m
%    By  Eran O. Ofek           November 1995
%--------------------------------------------------------------------
N1 = length(x1(:,1));
N2 = length(x2(:,1));
if N1~=N2,
   error('series must have the same length');
else
   N = N1;
end
c_x = 1;
c_y = 2;
if nargin==2,
   J   = floor(N/4);
elseif nargin==3,
   c_x = 1;
   c_y = 2;
   c_x = c_x;
else
   error('2 or 3 args only');
end
x1 = sortby(x1,c_x);
x2 = sortby(x2,c_x);
y1 = x1(:,c_y);
y2 = x2(:,c_y);
if J>N/2,
   'The lag is exceding size/2 of series'
end
my1  = mean(y1);
my2  = mean(y2);
MSy1 = y1 - my1;
MSy2 = y2 - my2;
Smy1 = std(MSy1);
Smy2 = std(MSy2);

for k=0:J,
   s = 0;
   for t=1:N-k,
      s = s + MSy1(t).*MSy2(t+k);
   end
   c(k+1) = (s./(N-k))./(Smy1.*Smy2);
end
Space = x1(2,c_x) - x1(1,c_x); 
rk = [Space.*rot90(0:J,3), rot90(c,3)];
Q = N.*sum(rk(1:J,2).*rk(1:J,2));
S = (1-rk(:,2).^2)./sqrt(N-1);
                                                                                                                                                                                                                                                                                                                                                                                               Contents.m                                                                                          0100644 0056337 0000144 00000012072 10316607147 011720  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  % ASTRONOMICAL - Time Series Utilities
%                                 By : Eran O. Ofek
%                                 Version: September 2005
% List of MATLAB programs in the timeseries package
%
% acf       - Autocorrelation function for evenly spaced, one dimensional
%             time series.
% afoevfor  - formating AFOEV variable star data file into a MATLAB
%             variable.
% arp       - model a time series by autoregressive process of order p.
% bin_by_eye- Plot data and define binning by eye.
%             The user mark (with the mouse) the beginning
%             and end points of each bin.
%             The left and right limits of each bin defined
%             by the user are marked by cyan and red
%             dashed lines, respectively.
% binn      - Binning a set of observations by equal number
%             of observations within each bin.
%             The program returns a matrix containing
%             the mean "time", mean value and value standard
%             deviation.
%             If the number of observations is not divided by
%             the number of points in each bin without
%             a reminder, then the last remaining observations
%             will not be used.
% ccf       - Cross correlation function for two, one dimensional time series.
%             Use Edelson & Krolik binning method for not-equaly spaced series.
% ccf_o     - Cross correlation function for evenly spaced two one
%             dimensional series.
% cosbell   - cosine bell function. Generating cosine bell function
%             in the range Start to End with its inner PercentFlat part
%             as flat function.
% curvlen   - calculate the length of curve by summing the distance
%             between successive points.
% cusum     - cumulative sum (CUSUM) chart for detecting non
%             stationarity in a series mean.
% equeliz   - Given two matrices [JD, Mag, ...], select all the
%             observations in the first matrix that was made in
%             the same instant (+/-threshold) and return the in
%             each line observations from both matrices that
%             was made at the same instant.
% extracph  - Extract observation within a given phase range.
% fitexp    - LSQ fitting of exponent model, to set of data points.
% fitgauss  - linear least squars gaussian fit to data.
% fitharmo  - LSQ harmonies fitting. Fit simultaneously any number of
%             frequncies, with any number of harmonics and linear terms.
% fitharmonw- LSQ harmonies fitting, with no errors (Weights=1).
% fitlegen  - LSQ Legendre polynomial fitting.
% fitpoly   - LSQ polynomial fitting.
% fitslope  - LSQ polynomial slope fitting (no a_0 term).
% fmaxs     - Given a matrix, find local maxima (in one of
%             the columns) and return the maxima position and height.
% folding   - Folding a set of observations into a period.
%             For each observation return the phase of the
%             observation within the period.
% hjd       - Convert Julian Day (UTC) to Helicentric/Barycentric
%             Julian Day (for geocentric observer).
% minclp    - Search for periodicity in a time series,
%             using the minimum-curve-length method.
%             The program calculates the curve length for
%             each trail frequency, and return the curve
%             length as function of frequency.
% pdm       - phase dispersion minimization.
% pdm_phot  - Phase dispersion minimazation of photon arrival time series.
% periodia  - classical periodigram calculating. normalization by
%             the variance of the data.
% periodis  - calculating a power spectrum to set of observations by
%             the method of Scargle.
% periodit  - calculating power spectrum as function of time.
% perioent  - periodicity search by minimizing the entropy.
% phot_event_me - Searching periodicity in time-tagged
%             events using information entropy.
%             For each trail period, the phase-magnitude space
%             is divided into m by m cells, and the information
%             entropy is calculated.
% poisson_event - Given a vector of time-tagged events, compare
%             the delta-time between sucssive events
%             with the exponential distribution.
% polysubs  - Subtract polynomial from a data set (no errors).
% runderiv  - Calculate the runing derivative of an unevenly spaced time
%             series, with flat weighting function
%             (e.g., the slope in each window).
%             Take into account slope-error and \chi^2.
% runmean   - Calculate the runing mean of an unevenly spaced time
%             series with different weight functions and weight scheme.
% sf_interp - Interpolation with structure function error propagation.
%             The error bar in each interpolated point is
%             calculated by adding in quadrature the
%             the error of the neastest point with the
%             amplitude of the stracture function at
%             the the lag equal to the difference between
%             the interpolated point and the nearest point.
% specwin   - spectral window of a time series.
%







                                                                                                                                                                                                                                                                                                                                                                                                                                                                      cosbell.m                                                                                           0100644 0056337 0000144 00000002634 07703100024 011535  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [CosB,Range]=cosbell(PrecentFlat,Range);
%--------------------------------------------------------------------
% cosbell function     cosine bell function
%                    Generating cosine bell function in the range
%                    [Start:End] with its inner PercentFlat part
%                    as flat function.
% input  : - The precentage of flat part of the cosine bell.
%            (in the range 0..1).
%          - Column vector of independent variable,
%            default is [0:0.01:1]'.
% output : - Column vector of cosine bell value.
%          - The Range (independent variable) column vector.
% Tested : Matlab 5.0
%     By : Eran O. Ofek           December 1997
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------

if nargin==1,
   Range = [0:0.01:1]';
elseif nargin>2,
   error('Number of input arguments should be 1 or 2');
end

[N,M]   = size(Range);
Start   = min(Range);
End     = max(Range);
Total   = End - Start;
TotalCB = Total.*(1 - PrecentFlat).*0.5;
StartCB = Start + Total.*(1 - PrecentFlat).*0.5;
EndCB   = End   - Total.*(1 - PrecentFlat).*0.5;

CosB    = zeros(N,1);
for Ind=1:1:N,
   if Range(Ind)<StartCB,
      CosB(Ind) = 0.5+0.5.*cos(pi.*(Range(Ind)-Start)./TotalCB + pi);
   elseif Range(Ind)>EndCB,
      CosB(Ind) = 0.5+0.5.*cos(pi.*(Range(Ind)-EndCB)./TotalCB);
   else
      CosB(Ind) = 1;
   end
end
                                                                                                    curvlen.m                                                                                           0100644 0056337 0000144 00000001547 07703311503 011600  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function l=curvlen(x,c_x,c_y);
%--------------------------------------------------------------------
% curvlen function    calculate the length of curve by summing 
%                   the distance between successive points.
% input  : - Matrix of observations, sorted by the column "c_x".
%          - c_x, column number of dependent variable, defualt is 1.
%          - c_y, column number of independent variable,  defualt is 2.
% output : - Curve length.
% Tested : Matlab 4.2
%     By : Eran O. Ofek           November 1993
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
if nargin==1,
   c_x = 1;
   c_y = 2;
elseif nargin>3,
   error('1 or 3 args only');
elseif nargin==2,
   error('1 or 3 args only');
end
dx = diff(x(:,c_x));
dy = diff(x(:,c_y));
c = sqrt(dx.*dx + dy.*dy);
l = sum(c);
                                                                                                                                                         cusum.m                                                                                             0100644 0056337 0000144 00000002521 07703312547 011260  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [ck,sl]=cusum(x,L);
%--------------------------------------------------------------------
% cusum function     cumulative sum (CUSUM) chart.
%                  to detect non stationarity in a series mean.
% input  : - Cvenly space observations matrix [Index Value].
%          - L (used to estimate the spectral density at zero frequency).
%            (L need to be small compared to N).
%            default is floor(0.1*N).
%          - K (used to estimate the spectral density at zero frequency).
%            default is 1.
% output : - [k, cumulative sum up to k].
%          - False Alarm Probability.
% Reference : Koen & Lombard 1993 MNRAS 263, 287-308.
% Tested : Matlab 4.2
%     By : Eran O. Ofek           October 1994
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
c_x = 1;
c_y = 2;
N = length(x(:,c_x));
if nargin==1,
   L = floor(0.1.*N);
   K = 1;
elseif nargin==2,
   K = 1;
elseif nargin==3,
   c_x = c_x;
else
   error('1, 2 or 3 args only');
end
y   = x(:,c_y) - mean(x(:,c_y));
ck = [x(:,c_x), cumsum(y)];
% calculating the spectral density at zero frequency.
f1 = K./N;
f2 = L./N;
df = f1:(f2-f1)./(L-K):f2;
p  = periodia(x,f1,f2,df);
S0 = sum(p(:,2))./L;
% normalizing the CUSUM.
ck = ck./sqrt(N.*S0);
Dn = max(abs(ck(:,2)));
sl = 2.*exp(-2.*Dn.*Dn);
                                                                                                                                                                               equeliz.m                                                                                           0100644 0056337 0000144 00000002722 07707036172 011606  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [LC1e,LC2e]=equeliz(LC1,LC2,Thresh);
%--------------------------------------------------------------------
% equeliz function     Given two matrices [JD, Mag, ...],
%                    select all the observations in the first matrix
%                    that was made in the same instant
%                    (+/-threshold) and return the in each line
%                    observations from both matrices that was made
%                    at the same instant.
% Input  : - first observations matrix. [JD, Mag, ...]
%          - second observations matrix. [JD, Mag, ...]
%          - Time tereshold, default is 0.001
% Output : - new observation matrix, in which observations not appear in
%            the original second matrix are deleted.
%          - new observation matrix in which, observations not appear in
%            the original first matrix are deleted.
% Tested : Matlab 5.3 
%     By : Eran O. Ofek    February 1993   last update September 2000
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html 
%--------------------------------------------------------------------
if (nargin==2),
   Thresh = 0.001;
elseif (nargin==3),
   % no default
else
   error('Illegal number of input parameters');
end

Ncol = length(LC1(1,:));

% equilize the lists
J = 0;
for I=1:1:length(LC1(:,1)),
   [MinTD, MinTDI] = min(abs(LC1(I,1)-LC2(:,1)));
   if (MinTD<Thresh),
      J = J + 1;
      LC1e(J,1:Ncol) = LC1(I,1:Ncol);
      LC2e(J,1:Ncol) = LC2(MinTDI,1:Ncol);
   end
end
                                              extracph.m                                                                                          0100644 0056337 0000144 00000002370 07707036420 011741  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function OutMat=extracph(InMat,T0,Period,PhaseRange);
%--------------------------------------------------------------------
% extracph function    Extract observation made in a given phase.
%                    Select all points with a given phase constraint.
%                    The phase and epoch is defined by the user.
% Input  : - observations matrix in which first column is time and
%            second column is observed value.
%          - ephemeris start time, T0.
%          - ephemeris period.
%          - Vector of phase range: [From To].
% Output : - Observations matrix, containing only the observations
%            for which the phase restriction are fulfilled.
% Tested : Matlab 5.3
%     By : Eran O. Ofek           October 1994,  Revised July 2000
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: OutMat=extracph(InMat,0,6.7,[0.1 0.2]);
%--------------------------------------------------------------------
if (nargin~=4),
   error('Illegal number of input arguments');
end


PhTemp = (InMat(:,1) - T0)./Period;
Phase  = PhTemp - floor(PhTemp);

if (PhaseRange(2)>PhaseRange(1)),
   I = find(Phase>=PhaseRange(1) & Phase<=PhaseRange(2));
else
   I = find(Phase>=PhaseRange(1) | Phase<=PhaseRange(2));
end

OutMat = InMat(I,1:end);

                                                                                                                                                                                                                                                                        fitexp.m                                                                                            0100644 0056337 0000144 00000003511 07713247225 011424  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [NewPar,NewParErr,Chi2,Deg,Cov]=fitexp(X,Y,DelY);
%--------------------------------------------------------------------
% fitexp function       Exponential fitting function
%                     fit data to function of the form:
%                     Y = A * exp(-X./Tau)
% Input  : - Vector of independent variable.
%          - Vector of dependent variable.
%          - vector of errors ins dependent variable.
% Output : - vector of parameters [A,Tau]
%          - vector of errors in parameters [err(A),err(Tau)]
%          - Chi square
%          - Degrees of freedom
%          - Covariance matrix
% Tested : Matlab 5.1
%     By : Eran O. Ofek           November 1996
%                             Last update: June 1999
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
N   = length(X);   % number of observations
Deg =  N - 2;      % degrees of freedom

% building the H matrix
H = [ones(N,1), -X.*ones(N,1)];

% Linearize the problem:
% NewY = ln(Y) = ln(A) - X./Tau 
NewY    = log(Y);
NewYerr = DelY./Y;

% The Variance Matrix
V = diag(NewYerr.^2);

% inverse the V Matrix
InvV = inv(V);

% The Covariance Matrix
Cov = inv(H'*InvV*H);

% The parameter vector [ln(A); 1./Tau]
Par    = Cov*H'*InvV*NewY;
ParErr = sqrt(diag(Cov));

% Transformin Parameter vector to A and Tau.
NewPar    = [exp(Par(1)), 1./Par(2)];
NewParErr = [NewPar(1).*ParErr(1), ParErr(2).*NewPar(2).^2];

'Number of degree of freedom :', Deg
Resid = NewY - H*Par;
Chi2  = sum((Resid./NewYerr).^2);
'Chi square per deg. of freedom       : ',Chi2/Deg
'Chi square error per deg. of freedom : ',sqrt(2/Deg)


% plot the data + fit
errorxy([X,Y,DelY],[1,2,3],'o');
hold on;
Np = 100;
X=[min(X):(max(X)-min(X))./(Np-1):max(X)]';
NewH = [ones(Np,1), -X.*ones(Np,1)];
Yplot=NewH*Par;
plot(X,exp(Yplot),'r');

                                                                                                                                                                                       fitgauss.m                                                                                          0100644 0056337 0000144 00000007204 07717710576 011765  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [NewPar,NewPar_Err,Cov,Chi2,Freedom,Resid]=fitgauss(X,Y,DelY,SigClip);
%--------------------------------------------------------------------
% fitpoly function       LSQ Gaussian fitting.
%                      fit Gaussian function of the form:
%                      Y = A * exp(-0.5.*((X-X0)./s).^2)
%                      to set of N data points. Return the parameters,
%                      the errors on the parameters,
%                      the \chi^2, and the covariance matrix.
% Input  : - Column vector of the independent variable.
%          - Column Vector of the dependent variable.
%          - Vector of the std error in the dependent variable.
%            If only one value is given, the points
%            are taken to be with equal weight. and Std error
%            equal to the value given.
%          - Sigma-Clipping (default is NaN, for no clipping).
% Output : - Fitted parameters [A,X0,s]
%          - Fitted errors in the parameters [DA,DX0,Ds]
%          - The covariance matrix.
%          - Chi2 of the fit.
%          - Degrees of freedom.
%          - The Y axis residuals vector.
% Tested : Matlab 5.3
%     By : Eran O. Ofek           October 1996
%                    Last Update  January 2001
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
MaxNIter = 5;   % maximum number of sigma-clipping iterations
if (nargin<4),
   SigClip = NaN;
else
   % do nothing
end

Deg  = 3;
NewY = log(Y); 

N_X  = length(X);
N_Y  = length(Y);
N_DY = length(DelY);
if (N_X~=N_Y),
   error('X and Y must have the same length');
end
if (N_X~=N_DY),
   if (N_DY==1),
      % take equal weights
      if (DelY<=0),
         error('DelY must be positive');
      else
         DelY = DelY.*ones(N_X,1);
      end
   else
      error('Y and DelY must have the same length');
   end
end

Resid = zeros(size(DelY));
if (isnan(SigClip)),
   MaxNIter = 1;
end

Iter = 0;
while (Iter<MaxNIter & (max(abs(Resid)>DelY | Iter==0))),
   Iter = Iter + 1;

   % sigma clipping
   if (isnan(SigClip)),
      % do not sigma clip
   else
      SCInd = find((abs(Resid)./(SigClip.*DelY))<1);  % find non-outlayers
size(SCInd)
      X    = X(SCInd);
      Y    = Y(SCInd);
      NewY = NewY(SCInd);
      DelY = DelY(SCInd);

      N_X  = length(X);
      N_Y  = length(Y);
      N_DY = length(DelY);  
   end

   % degree of freedom
   Freedom = N_X - (Deg + 1);
   
   % building the H matrix
   H = zeros(N_X,Deg);
   H(:,1) = ones(N_X,1);
   for Ind=2:1:Deg,
      H(:,Ind) = X.^(Ind-1);
   end
   
   % building the Covariance matrix
   V = diag(DelY.^2);
   V
   % Old - Memory consuming
   Cov     = inv(H'*inv(V)*H);
   Par     = Cov*H'*inv(V)*NewY;
   Par_Err = sqrt(diag(Cov));

   NewPar        = zeros(3,1);
   NewPar_Err    = zeros(3,1);
   NewPar(3)     = sqrt(-1./(2.*Par(3)));
   NewPar(2)     = NewPar(3).^2.*Par(2);
   NewPar(1)     = exp(Par(1) + 0.5.*NewPar(2).^2./(NewPar(3).^2));
   NewPar_Err(3) = (0.5./sqrt(2)).*Par_Err(3).*abs(Par(3)).^(-1.5);
   NewPar_Err(2) = sqrt((2.*NewPar(3).*Par(2).*NewPar_Err(3)).^2 + ...,
                        (NewPar(3).^2.*Par_Err(2)).^2);
   NewPar_Err(1) = sqrt((NewPar(1).*Par_Err(1)).^2 + ...,
                        (NewPar(1).*NewPar(2).^2.*NewPar(3).^(-3).*NewPar_Err(3)).^2 + ...,
                        (NewPar(1).*NewPar(2).*NewPar_Err(2)./(NewPar(3).^2)).^2);

                         
   
   %'Number of degree of freedom :', Freedom
   Resid = NewY - H*Par;
   Chi2  = sum((Resid./DelY).^2);

   %Chi2/Freedom
   %sqrt(2/Freedom)
end




%errorxy([X,Y,DelY],[1 2 3],'.');
%hold on;
%plot(X,H*Par,'r');


fprintf(1,'\n Number of iterations : %d \n',Iter);


                                                                                                                                                                                                                                                                                                                                                                                            fitharmo.m                                                                                          0100644 0056337 0000144 00000020626 07713247532 011745  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [Par,Par_Err,Cov,Chi2,Freedom,Par1,Resid]=fitharmo(X,Y,DelY,Har,Deg,PlotPar,File);
%--------------------------------------------------------------------
% fitharmo function      LSQ harmonies fitting.
%                      fit harmonies of the form:
%                      Y= a_1*sin(w1*t)     + b_1*cos(w1*t)   +
%                         a_2*sin(2*w1*t)   + b_2*cos(2*w1*t) + ...
%                         a_n*sin(n_1*w1*t) + b_n*cos(n_1*w1*t) + ...
%                         c_1*sin(w2*t)     + d_1*cos(w2*t) + ...
%                         s_0 + s_1*t + ... + s_n.*t.^n_s
%                         (note that w is angular frequncy, w=2*pi*f,
%                          the program is working with frequncy "f").
%                      to a set of N data points. Return the parameters,
%                      the errors on the parameters,
%                      the Chi squars, and the covariance matrix.
% input  : - Column vector of the independent variable.
%          - Column Vector of the dependent variable.
%          - Vector of the std error in the dependent variable.
%            If only one value is given, the points
%            are taken to be with equal weight. and Std error
%            equal to the value given.
%          - matrix of harmonies to fit.
%            N*2 matrix, where N is the number of different frequncies.
%            Each row should contain two numbers, the first is the
%            frequency to fit and the second is the number of harmonies
%            of that frequncy to fit. If there is more then one row
%            then all the frequncies and their harmonics will be fitted
%            simoltanusly.
%          - Degree of polynomials to fit. (Default is 0).
%          - Vector of plot's control characters.
%            If argument is given then X vs. Y graph is plotted.
%            If equal to empty string (e.g. '') then plot X vs. Y
%            with red fitted function line and yellow circs for
%            the observations.
%            If one or two character are given then the first character
%            is for the observations sign and the second for the fitted
%            function line.
%            If third character is given then histogram of resdiual
%            is plotted. when the third character should contain the
%            number of bins.
%          - File name in which summary table of the fit will be written.
%            The summary table includes all information regarding the
%            fit parameters and Chi2 test.
% output : - Fitted parameters [a_1,b_1,...,a_n,b_n,c_1,d_1,...,s_0,...]
%            The order of the parameters is like the order of the
%            freqencies matrix, and then the constant + linear terms.
%          - Fitted errors in the parameters [Da_1,Db_1,...]
%          - The covariance matrix.
%          - Chi2 of the fit.
%          - Degrees of freedom.
%          - sine/cosine parameters in form od Amp. and phase (in fraction),
%            pairs of lines for [Amp, Amp_Err; Phase, Phase_Err]...
%            phase are given in the range [-0.5,0.5].
%          - The Y axis residuals vector.
% See also : fitharmonw.m
% Tested : Matlab 5.1
%     By : Eran O. Ofek                    May 1994
%                             Last Update  Mar 1999
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
if (nargin<5),
   Deg = 1;
end
N_X  = length(X);
N_Y  = length(Y);
N_DY = length(DelY);
if (N_X~=N_Y),
   error('X and Y must have the same length');
end
if (N_X~=N_DY),
   if (N_DY==1),
      % take equal weights
      if (DelY<=0),
         error('DelY must be positive');
      else
         DelY = DelY.*ones(N_X,1);
      end
   else
      error('Y and DelY must have the same length');
   end
end

% number of parameters
N_Pars = Deg+1+2.*sum(Har(:,2));

% degree of freedom
Freedom = N_X - N_Pars;

% the size of the harmonies matrix
[Srow_Har,Scol_Har] = size(Har);
if (Scol_Har~=2),
   error('Number of columns in the harmonic freq. should be two');
end

% building the H matrix
H = zeros(N_X,N_Pars);
Counter = 0;
for I=1:1:Srow_Har,
   % run over number of harmonic per frequncy
   for J=1:1:Har(I,2),
      Counter = Counter + 1;
      H(:,Counter) = sin(2.*pi.*Har(I,1).*J.*X);
      Counter = Counter + 1;
      H(:,Counter) = cos(2.*pi.*Har(I,1).*J.*X);
   end
end
% add the constant term
Counter = Counter + 1;
H(:,Counter) = ones(N_X,1);
% add the linear terms
for I=1:1:Deg,
   Counter = Counter + 1;
   H(:,Counter) = X.^I;
end

% building the Covariance matrix
V = diag(DelY.^2);

% Old - Memory consuming
Cov     = inv(H'*inv(V)*H);
Par     = Cov*H'*inv(V)*Y;
Par_Err = sqrt(diag(Cov));

%'Number of degree of freedom :', Freedom
Resid = Y - H*Par;
Chi2  = sum((Resid./DelY).^2);

%Chi2/Freedom
%sqrt(2/Freedom)

if (nargin>5),
   'Plot Data'
   % plot results
   length(PlotPar);
   if (length(PlotPar)==0),
      PlotPar(1) = 'o';
      PlotPar(2) = 'r';
   end
   if (length(PlotPar)==1),
      PlotPar(2) = 'r';
   end
   figure(1);
   plot(X,Y,PlotPar(1));
   hold on;

   % ------------------------------------ New
   NewX = [min(X):((max(X)-min(X))./500):max(X)]';
   % building the newH matrix
   NewH = zeros(length(NewX),N_Pars);
   Counter = 0;
   for I=1:1:Srow_Har,
      % run over number of harmonic per frequncy
      for J=1:1:Har(I,2),
         Counter = Counter + 1;
         NewH(:,Counter) = sin(2.*pi.*Har(I,1).*J.*NewX);
         Counter = Counter + 1;
         NewH(:,Counter) = cos(2.*pi.*Har(I,1).*J.*NewX);
      end
   end
   % add the constant term
   Counter = Counter + 1;
   NewH(:,Counter) = ones(length(NewX),1);
   % add the linear terms
   for I=1:1:Deg,
      Counter = Counter + 1;
      NewH(:,Counter) = NewX.^I;
   end
   %----------
   plot(NewX,NewH*Par,PlotPar(2));
   xlabel('X');
   ylabel('Y');
   hold off;
   if (length(PlotPar)==3),
      % plot histogram of residuals
      figure(2);
      [Hist_X,Hist_N]=realhist(sort(abs(Resid)),str2num(PlotPar(3)),[0,max(abs(Resid)).*1.0001]);
      bar(Hist_X,Hist_N);
      axis([0,max(abs(Resid)).*1.0001,0,max(Hist_N)+1]);
      xlabel('X');
      ylabel('Number');
   end
end

% write summary file
if (nargin==7),
   Fid = fopen(File,'w');
   fprintf(Fid,'         %s \n','fitharmo.m - summary file');
   fprintf(Fid,' %s \n',['Created by Eran O. Ofek   at : ',date]);
   fprintf(Fid,'\n');
   fprintf(Fid,'%s \n','F# H#     Frequncy    1/Frequncy        A       dA           B       dB           C       dC          Fi       dFi');

   F_Coun = 0;
   for I=1:1:Srow_Har,
      % run over number of harmonic per frequncy
      for J=1:1:Har(I,2),
         F_Coun = F_Coun + 1;
         C  = sqrt(Par(F_Coun).^2 + Par(F_Coun+1).^2);
         Fi = atan2(Par(F_Coun+1),Par(F_Coun));
         C_Err  = sqrt((Par(F_Coun).*Par_Err(F_Coun)+Par(F_Coun+1).*Par_Err(F_Coun+1))./C);
         Fi_Err = sqrt((Par_Err(F_Coun+1) + Par(F_Coun+1).*Par_Err(F_Coun)./Par(F_Coun))./((Par(F_Coun).^2+Par(F_Coun+1).^2)./Par(F_Coun))); 
         F_Line = [I,J,Har(I,1).*J,1./(Har(I,1).*J),Par(F_Coun),Par_Err(F_Coun),Par(F_Coun+1),Par_Err(F_Coun+1),C,C_Err,Fi,Fi_Err];
         fprintf(Fid,'%d  %d  %12.6f %12.6f   %10.5f %8.5f  %10.5f %8.5f  %10.5f %8.5f  %10.5f %8.5f\n',F_Line');
         F_Coun = F_Coun + 1;
      end
   end

   fprintf(Fid,'\n');
   fprintf(Fid,'%s \n','Linear Terms');
   fprintf(Fid,'%s \n','#D      Coef     Err');
   for I=0:1:Deg,
      F_Coun = F_Coun + 1;
      F_Line = [I, Par(F_Coun), Par_Err(F_Coun)];
      fprintf(Fid,'%d  %10.5f %8.5f \n',F_Line);
   end
   fprintf(Fid,'\n');
   fprintf(Fid,'\n');
   fprintf(Fid,'%s \n','  Fit quality:');
   fprintf(Fid,'%s ',    ['No. Deg. of Fredom : ']);
   fprintf(Fid,' %d \n',Freedom);
   fprintf(Fid,'%s',['              Chi2 : ']);
   fprintf(Fid,' %10.4f \n',Chi2);
   fprintf(Fid,'%s',['      Reduced Chi2 : ']);
   fprintf(Fid,' %10.4f \n',Chi2/Freedom);
   fprintf(Fid,'%s', ['   Cumulative Chi2 : ']);
   fprintf(Fid,' %6.4f \n',chi2cdf(Chi2,Freedom));

   fclose(Fid);
end



% calculating amplitude and phase
Nab = 2.*sum(Har(:,2));
for I=1:2:Nab-1,
   A   = Par(I);
   B   = Par(I+1);
   DA  = Par_Err(I);
   DB  = Par_Err(I+1);
   % calculate amplitude
   C   = sqrt(A.^2+B.^2);
   DC  = sqrt(((A.*DA).^2+(B.*DB).^2)./(A.^2+B.^2));  
   % calculate phase
   Ph  = atan2(B,A);
   DPh = sqrt((A.*DB).^2+(B.*DA).^2)./(A.^2+B.^2);
   % convert phase from radian to fraction
   Ph  = Ph./(2.*pi);
   DPh = DPh./(2.*pi);

   Par1(I,1)   = C;
   Par1(I,2)   = DC;
   Par1(I+1,1) = Ph;
   Par1(I+1,2) = DPh;
end
                                                                                                          fitharmonw.m                                                                                        0100644 0056337 0000144 00000007552 07713247444 012317  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [Par,Chi2,Freedom,Par1,Resid]=fitharmonw(X,Y,Har,Deg);
%--------------------------------------------------------------------
% fitharmo function     LSQ harmonies fitting, with no errors
%                      (Weights=1) fit harmonies of the form:
%                      Y= a_1*sin(w1*t)     + b_1*cos(w1*t)   +
%                         a_2*sin(2*w1*t)   + b_2*cos(2*w1*t) + ...
%                         a_n*sin(n_1*w1*t) + b_n*cos(n_1*w1*t) + ...
%                         c_1*sin(w2*t)     + d_1*cos(w2*t) + ...
%                         s_0 + s_1*t + ... + s_n.*t.^n_s
%                         (note that w is angular frequncy, w=2*pi*f,
%                          the program is working with frequncy "f").
%                      to set of N data points. return the parameters,
%                      the errors on the parameters,
%                      the Chi squars, and the covariance matrix.
% input  : - Column vector of the independent variable.
%          - Column Vector of the dependent variable.
%          - matrix of harmonies to fit.
%            N*2 matrix, where N is the number of different frequncies.
%            Each row should contain two numbers, the first is the
%            frequency to fit and the second is the number of harmonies
%            of that frequncy to fit. If there is more then one row
%            then all the frequncies and their harmonics will be fitted
%            simoltanusly.
%          - Degree of polynomials to fit. (Default is 0).
% output : - Fitted parameters [a_1,b_1,...,a_n,b_n,c_1,d_1,...,s_0,...]
%            The order of the parameters is like the order of the
%            freqencies matrix, and then the constant + linear terms.
%          - Chi2 of the fit. (assuming DelY=1)
%          - Degrees of freedom.
%          - sine/cosine parameters in form od Amp. and phase (in fraction),
%            pairs of lines for [Amp, Amp_Err; Phase, Phase_Err]...
%            phase are given in the range [-0.5,0.5].
%          - The Y axis residuals vector. [calculated error for Chi^2=1,
%            can be calculated from mean(abs(Resid)) ].
% See also : fitharmo.m
% Tested : Matlab 5.3
%     By : Eran O. Ofek                 May 1994
%                          Last Update  August 2000
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
if (nargin<4),
   Deg = 1;
end
N_X  = length(X);
N_Y  = length(Y);
if (N_X~=N_Y),
   error('X and Y must have the same length');
end


% number of parameters
N_Pars = Deg+1+2.*sum(Har(:,2));

% degree of freedom
Freedom = N_X - N_Pars;

% the size of the harmonies matrix
[Srow_Har,Scol_Har] = size(Har);
if (Scol_Har~=2),
   error('Number of columns in the harmonic freq. should be two');
end

% building the H matrix
H = zeros(N_X,N_Pars);
Counter = 0;
for I=1:1:Srow_Har,
   % run over number of harmonic per frequncy
   for J=1:1:Har(I,2),
      Counter = Counter + 1;
      H(:,Counter) = sin(2.*pi.*Har(I,1).*J.*X);
      Counter = Counter + 1;
      H(:,Counter) = cos(2.*pi.*Har(I,1).*J.*X);
   end
end
% add the constant term
Counter = Counter + 1;
H(:,Counter) = ones(N_X,1);
% add the linear terms
for I=1:1:Deg,
   Counter = Counter + 1;
   H(:,Counter) = X.^I;
end


Par = H\Y;


%'Number of degree of freedom :', Freedom
Resid = Y - H*Par;
Chi2  = sum((Resid./1).^2);

%Chi2/Freedom
%sqrt(2/Freedom)


% calculating amplitude and phase
Nab = 2.*sum(Har(:,2));
for I=1:2:Nab-1,
   A   = Par(I);
   B   = Par(I+1);
   %DA  = Par_Err(I);
   %DB  = Par_Err(I+1);
   % calculate amplitude
   C   = sqrt(A.^2+B.^2);
   %DC  = sqrt(((A.*DA).^2+(B.*DB).^2)./(A.^2+B.^2));  
   % calculate phase
   Ph  = atan2(B,A);
   %DPh = sqrt((A.*DB).^2+(B.*DA).^2)./(A.^2+B.^2);
   % convert phase from radian to fraction
   Ph  = Ph./(2.*pi);
   %DPh = DPh./(2.*pi);

   Par1(I,1)   = C;
   %Par1(I,2)   = DC;
   Par1(I+1,1) = Ph;
   %Par1(I+1,2) = DPh;
end
                                                                                                                                                      fitlegen.m                                                                                          0100644 0056337 0000144 00000006760 07713247654 011741  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [Par,Par_Err,Cov,Chi2,Freedom,Resid]=fitlegen(X,Y,DelY,Deg,PlotPar);
%--------------------------------------------------------------------
% fitlegen function       LSQ Legendre polynomial fitting.
%                      fit Legendre polynomial of the form:
%                      Y= a_0*L_0(X) + a_1*L_1(X) +...+ a_n*L_n(X)
%                      to set of N data points. Return the parameters,
%                      the errors on the parameters,
%                      the Chi squars, and the covariance matrix.
% Input  : - Column vector of the independent variable.
%          - Column Vector of the dependent variable.
%          - Vector of the std error in the dependent variable.
%            If only one value is given then, points
%            are taken to be with equal weight. and Std error
%            equal to the value given.
%          - Degree of Legendre polynomial. (Default is 1).
%          - Vector of plot's control characters.
%            If argument is given then X vs. Y graph is plotted.
%            If equal to empty string (e.g. '') then plot X vs. Y
%            with red fitted function line and yellow circs for
%            the observations.
%            If one or two character are given then the first character
%            is for the observations sign and the second for the fitted
%            function line.
%            If third character is given then histogram of resdiual
%            is plotted. when the third character should contain the
%            number of bins.
% Output : - Fitted parameters [a0,a1,...]
%          - Fitted errors in the parameters [Da0,Da1,...]
%          - The covariance matrix.
%          - Chi2 of the fit.
%          - Degrees of freedom.
%          - The Y axis residuals vector.
% Tested : Matlab 5.3
%     By : Eran O. Ofek                  June 1998
%                           Last Update  June 1998
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
if (nargin<4),
   Deg = 1;
end
N_X  = length(X);
N_Y  = length(Y);
N_DY = length(DelY);
if (N_X~=N_Y),
   error('X and Y must have the same length');
end
if (N_X~=N_DY),
   if (N_DY==1)
      % take equal weights
      DelY = DelY.*ones(N_X,1);
   else
      error('Y and DelY must have the same length');
   end
end

% degree of freedom
Freedom = N_X - (Deg + 1);

% building the H matrix
AssocLegD = 1; % (associated Legendre of order 0)
H = zeros(N_X,Deg+1);
H(:,1) = legendre(0,X);
for Ind=2:1:Deg+1,
   TempLegen = legendre(Ind,X);
   H(:,Ind) = TempLegen(AssocLegD,:)';
end

% building the Covariance matrix
V = diag(DelY.^2);

% Old - Memory consuming
Cov     = inv(H'*inv(V)*H);
Par     = Cov*H'*inv(V)*Y;
Par_Err = sqrt(diag(Cov));


%'Number of degree of freedom :', Freedom
Resid = Y - H*Par;
Chi2 = sum((Resid./DelY).^2);

%Chi2/Freedom
%sqrt(2/Freedom)

%plot(X,Y)
%hold on;
%plot(X,H*Par,'o');

if (nargin==5),
   % plot results
   length(PlotPar);
   if (length(PlotPar)==0),
      PlotPar(1) = 'o';
      PlotPar(2) = 'r';
   end
   if (length(PlotPar)==1),
      PlotPar(2) = 'r';
   end
   figure(1);
   plot(X,Y,PlotPar(1));
   hold on;
   plot(X,H*Par,PlotPar(2));
   xlabel('X');
   ylabel('Y');
   hold off;
   if (length(PlotPar)==3),
      % plot histogram of residuals
      figure(2);
      [Hist_X,Hist_N]=realhist(sort(abs(Resid)),str2num(PlotPar(3)),[0,max(abs(Resid)).*1.0001]);
      bar(Hist_X,Hist_N);
      axis([0,max(abs(Resid)).*1.0001,0,max(Hist_N)+1]);
      xlabel('X');
      ylabel('Number');
   end
end




                fitpoly.m                                                                                           0100644 0056337 0000144 00000012340 07713247764 011623  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [Par,Par_Err,Cov,Chi2,Freedom,Resid]=fitpoly(X,Y,DelY,Deg,SigClip,PlotPar);
%--------------------------------------------------------------------
% fitpoly function       LSQ polynomial fitting.
%                      fit polynomial of the form:
%                      Y= a_0 + a_1*X + a_2*X^2 +...+ a_n*X^n
%                      to set of N data points. return the parameters,
%                      the errors on the parameters,
%                      the Chi squars, and the covariance matrix.
% Input  : - Column vector of the independent variable.
%          - Column Vector of the dependent variable.
%          - Vector of the std error in the dependent variable.
%            If only one value is given, the points
%            are taken to be with equal weight. and Std error
%            equal to the value given.
%            If two columns are given then the second column is taken
%            as the error in the independent variable,
%            and the problem is solve iteratively, starting with b=0.
%            (Num. Rec. chapter 15).
%          - Degree of polynomial. (Default is 1).
%          - Sigma-Clipping (default is NaN, for no clipping).
%          - Vector of plot's control characters.
%            If argument is given then X vs. Y graph is plotted.
%            If equal to empty string (e.g. '') then plot X vs. Y
%            with red fitted function line and yellow circs for
%            the observations.
%            If one or two character are given then the first character
%            is for the observations sign and the second for the fitted
%            function line.
%            If third character is given then histogram of resdiual
%            is plotted. when the third character should contain the
%            number of bins.
% Output : - Fitted parameters [a0,a1,...]
%          - Fitted errors in the parameters [Da0,Da1,...]
%          - The covariance matrix.
%          - Chi2 of the fit.
%          - Degrees of freedom.
%          - The Y axis residuals vector.
% Tested : Matlab 5.3
%     By : Eran O. Ofek                  March 1995
%                           Last Update  September 1999
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
MaxNIter = 5;   % maximum number of sigma-clipping iterations
if (nargin<4),
   Deg = 1;
   SigClip = NaN;
elseif (nargin<5),
   SigClip = NaN;
else
   % do nothing
end

N_X  = length(X);
N_Y  = length(Y);
N_DY = size(DelY,1);
N_DX = size(DelY,2);
if (N_DX==2),
   DelX = DelY(:,2);
   DelY = DelY(:,1);
else
   DelX = zeros(N_DY,1);
end
if (N_X~=N_Y),
   error('X and Y must have the same length');
end
if (N_X~=N_DY),
   if (N_DY==1),
      % take equal weights
      if (DelY<=0),
         error('DelY must be positive');
      else
         DelY = DelY.*ones(N_X,1);
      end
   else
      error('Y and DelY must have the same length');
   end
end

Resid = zeros(size(DelY));
if (isnan(SigClip)),
   MaxNIter = 1;
end

Iter = 0;
while (Iter<MaxNIter & (max(abs(Resid)>DelY | Iter==0))),
   Iter = Iter + 1;

   % sigma clipping
   if (isnan(SigClip)),
      % do not sigma clip
   else
      SCInd = find((abs(Resid)./(SigClip.*DelY))<1);  % find non-outlayers
      X    = X(SCInd);
      Y    = Y(SCInd);
      DelY = DelY(SCInd);
      DelX = DelX(SCInd);
      
      N_X  = length(X);
      N_Y  = length(Y);
      N_DY = length(DelY);  
   end

   % degree of freedom
   Freedom = N_X - (Deg + 1);
   
   % building the H matrix
   H = zeros(N_X,Deg+1);
   H(:,1) = ones(N_X,1);
   for Ind=2:1:Deg+1,
      H(:,Ind) = X.^(Ind-1);
   end
   
   % building the Covariance matrix
   B       = 0;
   DelXY   = sqrt(DelY.^2 + (B.*DelX).^2);
   V       = diag(DelXY.^2);
   
   % Old - Memory consuming
   Cov     = inv(H'*inv(V)*H);
   Par     = Cov*H'*inv(V)*Y;
   Par_Err = sqrt(diag(Cov));
   
   
   %'Number of degree of freedom :', Freedom
   Resid = Y - H*Par;
   Chi2  = sum((Resid./DelXY).^2);

   %Chi2/Freedom
   %sqrt(2/Freedom)
   
   NB_Iter = 1;
   while (abs(B-Par(2))>Par_Err(2).*1e-8),
      % iterate with DelX (B~=0)
      B = Par(2);   
      % building the Covariance matrix
      DelXY   = sqrt(DelY.^2 + (B.*DelX).^2);
      V       = diag(DelXY.^2);
      % Old - Memory consuming
      Cov     = inv(H'*inv(V)*H);
      Par     = Cov*H'*inv(V)*Y;
      Par_Err = sqrt(diag(Cov));
      %'Number of degree of freedom :', Freedom
      Resid = Y - H*Par;
      Chi2  = sum((Resid./DelXY).^2);
      NB_Iter = NB_Iter + 1;
   end   
   
end

if (nargin==6),
   % plot results
   length(PlotPar);
   if (length(PlotPar)==0),
      PlotPar(1) = 'o';
      PlotPar(2) = 'r';
   end
   if (length(PlotPar)==1),
      PlotPar(2) = 'r';
   end
   figure(1);
   plot(X,Y,PlotPar(1));
   hold on;
   plot(X,H*Par,PlotPar(2));
   xlabel('X');
   ylabel('Y');
   hold off;
   if (length(PlotPar)==3),
      % plot histogram of residuals
      figure(2);
      [Hist_X,Hist_N]=realhist(sort(abs(Resid)),str2num(PlotPar(3)),[0,max(abs(Resid)).*1.0001]);
      bar(Hist_X,Hist_N);
      axis([0,max(abs(Resid)).*1.0001,0,max(Hist_N)+1]);
      xlabel('X');
      ylabel('Number');
   end
end


%errorxy([X,Y,DelY],[1 2 3],'.');
%hold on;
%plot(X,H*Par,'r');


fprintf(1,'\n Number of iterations : %d \n',Iter);


                                                                                                                                                                                                                                                                                                fitpow.m                                                                                            0100644 0056337 0000144 00000004171 07713250035 011431  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [NewPar,NewParErr,Chi2,Deg,Cov,Resid]=fitpow(X,Y,DelY);
%--------------------------------------------------------------------
% fitpow function       Power law fitting function
%                     fit data to function of the form:
%                     Y = A * X ^(Gamma)
% input  : - Vector of independent variable.
%          - Vector of dependent variable.
%          - vector of errors ins dependent variable.
%            if scalar is given then its taken
%            as equal error.
% output : - vector of parameters [A,Gamma]
%          - vector of errors in parameters [err(A),err(Gamma)]
%          - Chi square
%          - Degrees of freedom
%          - Covariance matrix
%          - The Y axis residuals vector. [calculated error for Chi^2=1,
%            can be calculated from mean(abs(Resid)) ].
% Tested : Matlab 5.0
%     By : Eran O. Ofek             November 1996
%                               last update: June 1999
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
N   = length(X);   % number of observations
Deg =  N - 2;      % degrees of freedom

% building the H matrix
H = [ones(N,1), log(X).*ones(N,1)];

% Linearize the problem:
% NewY = ln(Y) = ln(A) + Gamma * ln(X)

if (length(DelY)==1),
   DelY = DelY.*ones(size(X));
end
 
NewY    = log(Y);
NewYerr = DelY./Y;

% The Variance Matrix
V = diag(NewYerr.^2);

% inverse the V Matrix
InvV = inv(V);

% The Covariance Matrix
Cov = inv(H'*InvV*H);

% The parameter vector [ln(A); 1./Tau]
Par    = Cov*H'*InvV*NewY;
ParErr = sqrt(diag(Cov));

% Transformin Parameter vector to A and Gamma.
NewPar    = [exp(Par(1)); Par(2)];
NewParErr = [NewPar(1).*ParErr(1); ParErr(2)];

'Number of degree of freedom :', Deg
Resid = NewY - H*Par;
Chi2  = sum((Resid./NewYerr).^2);
'Chi square per deg. of freedom       : ',Chi2/Deg
'Chi square error per deg. of freedom : ',sqrt(2/Deg)


Resid = NewY - H*Par;

% plot the data + fit
%errorxy([X,Y,DelY],[1,2,3],'o');
%hold on;
%Np = 100;
%X=[min(X):(max(X)-min(X))./(Np-1):max(X)]';
%NewH = [ones(Np,1), log(X).*ones(Np,1)];
%Yplot=NewH*Par;
%plot(X,exp(Yplot),'r');

                                                                                                                                                                                                                                                                                                                                                                                                       fitslope.m                                                                                          0100644 0056337 0000144 00000012305 07713250145 011746  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [Par,Par_Err,Cov,Chi2,Freedom,Resid]=fitslope(X,Y,DelY,Deg,SigClip,PlotPar);
%--------------------------------------------------------------------
% fitslope function       LSQ polynomial fitting.
%                      fit polynomial of the form:
%                      Y= a_1*X + a_2*X^2 +...+ a_n*X^n
%                      to set of N data points. Return the parameters,
%                      the errors on the parameters,
%                      the Chi squars, and the covariance matrix.
% Input  : - Column vector of the independent variable.
%          - Column Vector of the dependent variable.
%          - Vector of the std error in the dependent variable.
%            If only one value is given, the points
%            are taken to be with equal weight. and Std error
%            equal to the value given.
%            If two columns are given then the second column is taken
%            as the error in the independent variable,
%            and the problem is solve iteratively, starting with b=0.
%            (Num. Rec. chapter 15).
%          - Degree of polynomial. (Default is 1).
%          - Sigma-Clipping (default is NaN, for no clipping).
%          - Vector of plot's control characters.
%            If argument is given then X vs. Y graph is plotted.
%            If equal to empty string (e.g. '') then plot X vs. Y
%            with red fitted function line and yellow circs for
%            the observations.
%            If one or two character are given then the first character
%            is for the observations sign and the second for the fitted
%            function line.
%            If third character is given then histogram of resdiual
%            is plotted. when the third character should contain the
%            number of bins.
% Output : - Fitted parameters [a0,a1,...]
%          - Fitted errors in the parameters [Da0,Da1,...]
%          - The covariance matrix.
%          - Chi2 of the fit.
%          - Degrees of freedom.
%          - The Y axis residuals vector.
% Tested : Matlab 5.3
%     By : Eran O. Ofek                  March 1995
%                           Last Update  September 1999
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
MaxNIter = 5;   % maximum number of sigma-clipping iterations
if (nargin<4),
   Deg = 1;
   SigClip = NaN;
elseif (nargin<5),
   SigClip = NaN;
else
   % do nothing
end

N_X  = length(X);
N_Y  = length(Y);
N_DY = size(DelY,1);
N_DX = size(DelY,2);
if (N_DX==2),
   DelX = DelY(:,2);
   DelY = DelY(:,1);
else
   DelX = zeros(N_DY,1);
end
if (N_X~=N_Y),
   error('X and Y must have the same length');
end
if (N_X~=N_DY),
   if (N_DY==1),
      % take equal weights
      if (DelY<=0),
         error('DelY must be positive');
      else
         DelY = DelY.*ones(N_X,1);
      end
   else
      error('Y and DelY must have the same length');
   end
end

Resid = zeros(size(DelY));
if (isnan(SigClip)),
   MaxNIter = 1;
end

Iter = 0;
while (Iter<MaxNIter & (max(abs(Resid)>DelY | Iter==0))),
   Iter = Iter + 1;

   % sigma clipping
   if (isnan(SigClip)),
      % do not sigma clip
   else
      SCInd = find((abs(Resid)./(SigClip.*DelY))<1);  % find non-outlayers
      X    = X(SCInd);
      Y    = Y(SCInd);
      DelY = DelY(SCInd);
      DelX = DelX(SCInd);
      
      N_X  = length(X);
      N_Y  = length(Y);
      N_DY = length(DelY);  
   end

   % degree of freedom
   Freedom = N_X - (Deg);
   
   % building the H matrix
   H = zeros(N_X,Deg);
   H(:,1) = X;
   for Ind=2:1:Deg,
      H(:,Ind) = X.^Ind;
   end
   
   % building the Covariance matrix
   B       = 0;
   DelXY   = sqrt(DelY.^2 + (B.*DelX).^2);
   V       = diag(DelXY.^2);
   
   % Old - Memory consuming
   Cov     = inv(H'*inv(V)*H);
   Par     = Cov*H'*inv(V)*Y;
   Par_Err = sqrt(diag(Cov));
   
   
   %'Number of degree of freedom :', Freedom
   Resid = Y - H*Par;
   Chi2  = sum((Resid./DelXY).^2);

   %Chi2/Freedom
   %sqrt(2/Freedom)
   
   NB_Iter = 1;
   while (abs(B-Par(1))>Par_Err(1).*1e-8),
      % iterate with DelX (B~=0)
      B = Par(1);   
      % building the Covariance matrix
      DelXY   = sqrt(DelY.^2 + (B.*DelX).^2);
      V       = diag(DelXY.^2);
      % Old - Memory consuming
      Cov     = inv(H'*inv(V)*H);
      Par     = Cov*H'*inv(V)*Y;
      Par_Err = sqrt(diag(Cov))
      %'Number of degree of freedom :', Freedom
      Resid = Y - H*Par;
      Chi2  = sum((Resid./DelXY).^2);
      NB_Iter = NB_Iter + 1;
   end   
   
end

if (nargin==6),
   % plot results
   length(PlotPar);
   if (length(PlotPar)==0),
      PlotPar(1) = 'o';
      PlotPar(2) = 'r';
   end
   if (length(PlotPar)==1),
      PlotPar(2) = 'r';
   end
   figure(1);
   plot(X,Y,PlotPar(1));
   hold on;
   plot(X,H*Par,PlotPar(2));
   xlabel('X');
   ylabel('Y');
   hold off;
   if (length(PlotPar)==3),
      % plot histogram of residuals
      figure(2);
      [Hist_X,Hist_N]=realhist(sort(abs(Resid)),str2num(PlotPar(3)),[0,max(abs(Resid)).*1.0001]);
      bar(Hist_X,Hist_N);
      axis([0,max(abs(Resid)).*1.0001,0,max(Hist_N)+1]);
      xlabel('X');
      ylabel('Number');
   end
end


%errorxy([X,Y,DelY],[1 2 3],'.');
%hold on;
%plot(X,H*Par,'r');


fprintf(1,'\n Number of iterations : %d \n',Iter);


                                                                                                                                                                                                                                                                                                                           fmaxs.m                                                                                             0100644 0056337 0000144 00000004576 10257771315 011256  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [Res,Ind,WMeanPos]=fmaxs(Mat,ColX,ColY,MeanWindowSize);
%--------------------------------------------------------------------
% fmaxs function    Given a matrix, find local maxima (in one of
%                 the columns) and return the maxima position
%                 and height.
% Input  : - Matrix of at least two columns.
%          - The column index of the independent variable,
%            default is 1.
%          - The column index of the dependent variable,
%            default is 2.
%          - Half window size for calculating the peak position
%            using wmean, default is 3.
% Output : - Two column matrix of all local maxima.
%            The first column is the independent variable
%            (maxima position) while the second column is the
%            dependent variable (maxima height).
%          - Vector of indices of the local maxima in the original
%            matrix.
%          - Weighted mean position, error and maximum of fitted parabola
%            for each peak.
%            The weights are taken as 1/sqrt(Height).
% Tested : Matlab 4.2
%          Matlab 7.0
%     By : Eran O. Ofek           December 1993
%                                     June 2005
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
if (nargin==1),
   ColX           = 1;
   ColY           = 2;
   MeanWindowSize = 3;
elseif (nargin==2),
   ColY           = 2;
   MeanWindowSize = 3;
elseif (nargin==3),
   MeanWindowSize = 3;
elseif (nargin==4),
   % do nothing
else
   error('Illegal number of input arguments');
end

DiffVec = diff(sign(diff([0;Mat(:,ColY);0])));
Ind     = find(DiffVec==-2);

Res     = Mat(Ind,[ColX, ColY]);
LenMat  = size(Mat,1);

if (nargout>2),
   N  = length(Ind);   % number of peaks
   WMeanPos = zeros(N,3);
   for I=1:1:N,
      if ((Ind(I)-MeanWindowSize)<1 | (Ind(I)+MeanWindowSize)>LenMat),
         % can't calculate wmean, set to NaN
         Mean        = NaN;
         Err         = NaN;
         MaxParabola = NaN;
      else
         % Weighted mean
         SubMat     = Mat([Ind(I)-MeanWindowSize:1:Ind(I)+MeanWindowSize],:);
         [Mean,Err] = wmean([SubMat(:,1),1./sqrt(SubMat(:,2))]);

         % fit parabola
         Par = polyfit(SubMat(:,1),SubMat(:,2),2);
         MaxParabola = -Par(2)./(2.*Par(1));
      end
      WMeanPos(I,:) = [Mean, Err, MaxParabola];
   end
end
                                                                                                                                  folding.m                                                                                           0100644 0056337 0000144 00000002406 07713255511 011546  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function y=folding(x,p,c);
%--------------------------------------------------------------------
% folding function     Folding a set of observations into a period.
%                    For each observation return the phase of the
%                    observation within the period.
% Input  : - Matrix in which one of the columns is the time.
%            The folding is done by the time column.
%          - period to fold into.
%          - column number to fold by (time column). Defualt is 1.
% output : - Matrix similar to the input matrix, but in which
%            the time column is replaced by the phase, and
%            the time column is concatenate in the last column.
%            The output matrix is sorted by phase.
% Tested : Matlab 3.5
%     By : Eran O. Ofek              November 1993
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
if nargin==2,
   c=1;
elseif nargin==3,
   % do nothing
else
   error('Illegal number of input arguments');
end

jd_col      = length(x(1,:)) + 1;
y           = zeros(length(x(:,1)),jd_col);
TEMP        = x(:,c);
r           = TEMP;
TEMP        = TEMP./p-floor(TEMP./p);
y           = x;
y(:,c:c)    = TEMP;
y(:,jd_col) = r;
y           = sortrows(y,c);
                                                                                                                                                                                                                                                          hjd.m                                                                                               0100644 0056337 0000144 00000004656 07713562354 010710  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [OutJD,ObjVel]=hjd(JD,ObjCoo,Type);
%---------------------------------------------------------------------------
% hjd function         Convert Julian Day (UTC) to Helicentric/Barycentric
%                    Julian Day (for geocentric observer).
% Input  : - Column vector of julian days (UTC time system).
%          - J2000.0 object coordinates, [RA, Dec], in radians.
%          - Observer Geodetic position,
%            [East_Long (rad), North_Lat (rad), Geodetic height (meters)].
%            If geodetic position is not given (or empty matrix),
%            then assume geocentric observer.
%          - Output type:
%            'lh' - low accuracy heliocentric JD.
%            'hh' - high accuracy (full VSOP87) heliocentric JD.
%            'hb' - high accuracy (full VSOP87) barycentric JD, default.
% Output : - Heliocentric/Barycentric julian day (for geocentric observer).
%          - Heliocentric/Barycentric velocity comonent in object
%            direction [km/sec] (only for 'hh' | 'hb' options).
% Example : [OutJD,ObjVel]=hjd(JD,[RA Dec],'hb');
% Tested : Matlab 5.3 
%     By : Eran O. Ofek                  November 1993
%                          Last update : August 2003
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%---------------------------------------------------------------------------
  SEC_IN_DAY = 86400.0;
if (nargin==2),
   Type = 'hb';
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end

C  = get_constant('c','agd');
AU = get_constant('au','SI')./1000;
N  = length(JD);

switch Type
 case 'lh'
    [RA,Dec]     = suncoo(JD,'j');
    Coo          = cosined([RA+pi, -Dec]).';
    Vel          = [NaN; NaN; NaN];
    Vel          = repmat(Vel,1,N);
 case 'hh'
    [DeltaT,DUT] = delta_t(JD);
    JD_TT        = JD + (DUT + DeltaT)./SEC_IN_DAY; 
    [Coo,Vel]    = calc_vsop87(JD_TT, 'Earth', 'a', 'E');

 case 'hb'
    [DeltaT,DUT] = delta_t(JD);
    JD_TT        = JD + (DUT + DeltaT)./SEC_IN_DAY; 
    [Coo,Vel]    = calc_vsop87(JD_TT, 'Earth', 'e', 'E');

 otherwise
    error('Unknown Type option');
end

ObjPos = [cosined(ObjCoo)].';


DelJD = zeros(N,1);
ObjVel = zeros(N,1);
for I=1:1:N,
   EarthCoo = Coo(:,I);
   EarthVel = Vel(:,I);

   DelJD(I)  = norm(EarthCoo).*dot(ObjPos./norm(ObjPos),EarthCoo./norm(EarthCoo))./C;
   ObjVel(I) = norm(EarthVel).*dot(ObjPos./norm(ObjPos),EarthVel./norm(EarthVel)).*AU./SEC_IN_DAY;
end

OutJD = JD - DelJD;

DelJD*86400
%ObjVel


                                                                                  minclp.m                                                                                            0100644 0056337 0000144 00000003424 07713564223 011412  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [pl,m]=minclp(x,mn,mx,int,bin,c_x,c_y);
%--------------------------------------------------------------------
% minclp function        Search for periodicity in a time series,
%                      using the minimum-curve-length method.
%                      The program calculates the curve length for
%                      each trail frequency, and return the curve
%                      length as function of frequency.
% Input  : - Data matrix, sorted by time.
%          - Minimum frequency to search.
%          - Maximum frequency to search.
%          - Frequency interval to search. defualt is 0.2/Time_Span.
%          - Optional binning. If 0, then don't use binning, if ~=0
%            then use bins of size 'bin'. Default is 0.
%          - The time column, defualt is 1.
%          - The dependent variable column, defualt is 2.
% Output : - Curve length as function of frequency,
%            [Frequency, Lengh].
%          - Frequency for which minimum length was obtained.
% Tested : Matlab 4.2
%     By : Eran O. Ofek           November 1993
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
if nargin==3,
   c_x = 1;
   c_y = 2;
   bin = 0;
   int = 0.2./(max(x(:,c_x)) - min(x(:,c_x)))
elseif nargin==4,
   c_x = 1;
   c_y = 2;
   bin = 0;
elseif nargin==5,
   c_x = 1;
   c_y = 2;
elseif nargin==6,
   c_y = 2;
elseif nargin==7,
   % do nothing
else
   error('Illegal number of input arguments');
end


if bin>0.9,
   error('bin>0.9, bin is in phase');
end
j = 1;
for freq=mn:int:mx,
   f = folding(x,1./freq,c_x);
   if bin~=0,
      [f,tem1,tem2] = bining(f,bin,c_x);
   end
   pl(j,1) = k;
   pl(j,2) = curvlen(f,c_x,c_y);
   k = k + int;
   j = j + 1;
end
[ml,ind] = min(pl(:,2:2));
m = pl(ind,1);
                                                                                                                                                                                                                                            pdm.m                                                                                               0100644 0056337 0000144 00000003044 07713542035 010703  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function PDM=pdm(Data,FreqVec,BinNum);
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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            pdm_phot.m                                                                                          0100644 0056337 0000144 00000005174 07713542232 011742  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [F,Variance,StD_Mean]=pdm_phot(X,Nb,Freq_Vec);
%--------------------------------------------------------------------
% pdm_phot function    Phase Dispersion Minimization to Photon
%                    arrival time data.
% Input  : - Sorted vector of arrival times.
%          - Number of bins.
%          - Vector of frequencies to search,
%            [Low Frequency, High Frequency, Frequency Interval]
%            default is to choose the Low Freq. as 1/(Time Span)
%            the High Freq. as the mean(diff(photon arrival time))
%            Freq. Interval as 1/(4 X Time Span)
% Output : - Vector of frequencies
%          - Vector of variances coresponding to each trial frequency.
%          - Vector of standard deviations of means coresponding
%            to each trial frequency.
% Tested : Matlab 5.1
%     By : Eran O. Ofek           November 1996
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
if nargin==2,
   Time_Span = Max(X) - Min(X);
   Low_Freq  = 1./(Time_Span);
   High_Freq = mean(diff(X));
   Freq_Int  = 1./(4.*Time_Span);
elseif nargin==3,
   % do nothing
else
   error('Illegal number of input arguments');
end


% assuming X is sorted
%X = sort(X)

% Number of Photons
NoP = length(X);


Low_Freq  = Freq_Vec(1);
High_Freq = Freq_Vec(2);
Freq_Int  = Freq_Vec(3);


% calculating Variance for all the data
Diff_All = diff(X);
Vari_All = sum((Diff_All - mean(Diff_All)).^2)./(NoP - 2);

% initialize Vector of Indices
%VI = zeros(1,Nb-1);
% initialize Vector of StD
StD_V = zeros(1,Nb);
% initialize Vector of Means
Mean_V = zeros(1,Nb);
% initialize Vector of Number of Photons in Bin
NoPB_V = zeros(1,Nb);
% initialize Vector of Variance
Variance = zeros(1,length(Low_Freq:Freq_Int:High_Freq));
% initialize Vector of StD of Means 
StD_Mean = zeros(1,length(Low_Freq:Freq_Int:High_Freq));
% initialize Vector of Frequencies of Means 
F = [Low_Freq:Freq_Int:High_Freq];


J = 1;
for Freq=Low_Freq:Freq_Int:High_Freq,
   F(J)      = Freq;
   P         = X.*Freq - floor(X.*Freq);
   Sorted_P  = sort(P);
   Diff_P    = diff(Sorted_P);

   First_Ind = 1;
   for Bin_N=1:1:Nb-1,
      Bin_Ind       = bin_sear(Sorted_P, Bin_N./Nb);
      StD_V(Bin_N)  = std(First_Ind:Bin_Ind);
      Mean_V(Bin_N) = mean(First_Ind:Bin_Ind);
      NoPB_V(Bin_N) = Bin_Ind -  First_Ind + 1;

      First_Ind     = Bin_Ind
   end
   StD_V(Nb)  = std(First_Ind:NoP);
   Mean_V(Nb) = mean(First_Ind:NoP);
   NoPB_V(Nb) = NoP - First_Ind + 1;


   % calculate statistics in each bin
   Variance(J) = sum((NoPB_V - 1).*StD_V.^2./(sum(NoPB_V) - Nb);
   StD_Mean(J) = std(Mean_V);
end

   



                                                                                                                                                                                                                                                                                                                                                                                                    periodia.m                                                                                          0100644 0056337 0000144 00000004175 07713542347 011733  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [Pxw,Fm]=periodia(x,l_f,h_f,df,c_x,c_y);
%--------------------------------------------------------------------
% periodia function     Classical power-spectrum of a time series
%                     normalized the power by the variance of
%                     the data.
% input  : - Time series matrix, [Time, Mag], in which the first column
%            is the time and the second column is the magnitude.
%          - Lowest frequency (h_l).
%          - Highest frequency (h_f).
%          - Frequency interval (df).
%          - The column of time (c_x), default is 1.
%          - The column of magnitudes (c_y), default is 2.
% output : - Periodigram matrix. normalized with the variance 
%            of the observations (Horne & Baliunas 1986).
%            The first column is the frequency and
%            the second column is the power.
%          - The peaks of the periodogram sorted by the probability.
%            [Frequency, Power, Period, (1-False alarm probability)].
%            The probability is good only for evenly spaced data
% See Also : periodis; periodit; pdm; fitharmo
% Reference: Koen, C. 1990, ApJ 348, 700-702.
% Tested   : Matlab 4.2
%       By : Eran O. Ofek           December 1993
%      URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
if nargin==4,
   c_x = 1;
   c_y = 2;
elseif nargin==5,
   c_y = 2;
elseif nargin==6,
   % do nothing
else
   error('Illegal number of input arguments');
end

% set probability cutoff to zero.
pr = 0;
N0 = length(x(:,c_x));
Ni = -6.362 + 1.193.*N0 + 0.00098.*N0.*N0;
tem0 = x(:,c_y) - mean(x(:,c_y));
f_ind = l_f;
k = 1;
Pxw = zeros((h_f-l_f)./df,2);
while f_ind<h_f,
   Pxw(k,1) = f_ind; 
   temp = abs(sum(tem0.*exp(-i*2*pi*f_ind*x(:,c_x))));
   Pxw(k,2) = temp*temp/N0;
   f_ind = f_ind + df;
   k = k + 1;
end

% normalization of the periodogram
noise = std(x(:,c_y)).^2;
Pxw(:,2) = Pxw(:,2)./noise;


if (nargout==2),
   DiffVec = diff(sign(diff([0;Pxw(:,2);0])));
   IdV     = find(DiffVec==-2);

   Fm      = [Pxw(IdV,1), Pxw(IdV,2), 1./Pxw(IdV,1), (1-exp(-Pxw(IdV,2))).^Ni];
   Fm      = sortrows(Fm,2);
end
                                                                                                                                                                                                                                                                                                                                                                                                   periodis.m                                                                                          0100644 0056337 0000144 00000004614 10030377546 011744  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [Pxw, Fm] = periodis(x,l_f,h_f,df,pr);
%------------------------------------------------------------------------
% periodis function        Scargle periodigram
%                        calculate the Scargle power spectrum of a
%                        time series.
% input  : - Timeseries matrix, [Time, Mag], in which the first column
%            is the tim and the second column is the magnitude.
%          - Lowest frequency (h_l).
%          - Highest frequency (h_f).
%          - Frequency interval (df).
%          - The probability cutoff (pr), default no cutoff.
%            Power spectra peaks with probability smaller than this
%            cutoff are not listed.
% output : - Periodigram matrix. normalized with the variance 
%            of the observations (Horne & Baliunas 1986).
%            The first column is the frequency and
%            the second column is the power.
%          - The peaks of the periodogram sorted by the spectral power.
%            [Frequency, Power, Period, (1-False alarm probability)].
% See Also : periodia; periodit; pdm; fitharmo
% Reference: Scargle, J.D. ApJ 263, 835-853 (1982).
%	          Horne, J.H. & Baliunas, S.L. ApJ 302, 757-763 (1986).
% Tested   : Matlab 4.2
%       By : Eran O. Ofek           March 1994
%      URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%------------------------------------------------------------------------
c_x = 1;
c_y = 2;
if nargin==4,
   pr = 0; 
elseif nargin==5,
   % do nothing
else
   error('Illegal number of input arguments');
end


noise = std(x(:,c_y)).*std(x(:,c_y));
N0 = length(x(:,c_x));
Ni = -6.362 + 1.193.*N0 + 0.00098.*N0.*N0;
tem0 = x(:,c_y) - mean(x(:,c_y));
f_ind = l_f;
k = 1;
Pxw = zeros((h_f-l_f)./df,2);
while f_ind<h_f,
   Pxw(k,1) = f_ind; 
   om = 2.*pi.*f_ind;
   tau = sum(sin(2.*om.*x(:,c_x)))./sum(cos(2.*om.*x(:,c_x)));
   tau = atan(tau)./(2.*om);
   Axc = cos(om.*(x(:,c_x) - tau));
   Axs = sin(om.*(x(:,c_x) - tau));
   Ax1 = sum(tem0.*Axc);
   Ax1 = Ax1.*Ax1;
   Ax2 = sum(tem0.*Axs);
   Ax2 = Ax2.*Ax2;
   temp = Ax1./sum(Axc.*Axc) + Ax2./sum(Axs.*Axs);
   Pxw(k,2) = 0.5.*temp;
   f_ind = f_ind + df;
   k = k + 1;
end
% normalization of the periodogram
Pxw(:,2) = Pxw(:,2)./noise;



if (nargout==2),
   DiffVec = diff(sign(diff([0;Pxw(:,2);0])));
   IdV     = find(DiffVec==-2);

   Fm      = [Pxw(IdV,1), Pxw(IdV,2), 1./Pxw(IdV,1), (1-exp(-Pxw(IdV,2))).^Ni];
   Fm      = sortrows(Fm,2);
end
                                                                                                                    periodit.m                                                                                          0100644 0056337 0000144 00000015757 07734774521 011772  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [SuperP,SuperF,MidTime,FreqSer]=periodit(TimeSerMat,EpochPars,TaperPars,FreqVec,PeriodType,Disp);
%---------------------------------------------------------------------
% periodit function    Calculate periodogram as function of
%                    time. This script can be used for searchnig
%                    for long term varibility in periodicities.
%                    The time span is divided into several
%                    epochs with or without overlays. Each epoch
%                    could be tapered, and periodogram (Clasical or
%                    Scargle) is calculated. The results are displaied
%                    as a mesh of frequncy vs. time vs. power.
% Input  : - Matrix of observations. Should contain at least two
%            column. First column for time and second column for
%            "magnitude". Each row for each observation.
%          - vector of parameters defined the epochs of sub
%            periodograms. First element should contain the number
%            of epochs in the time span. and the second element
%            should contain the size of each sub epoch in units of
%            the total time span. default is [10,0.2].
%          - Tapering function parameters. Each sub epoch could be
%            weighted by a tapering function.
%            Avaliable tapers:
%            ['C',p] : cosine bell taper while "p" is the precentage
%                       of flat (uniform) part of the cosine bell in
%                       the range 0-1.
%            ['F']   : Flat (uniform) taper. No tapering.
%            default is ['C',0.6].
%          - frequency vector of the type:
%            [Low_Freqency, High_Frequency, Frequency_Interval].
%            default is [0,1./(2min(Interval)),1./(2(Time_Span)].
%          - Periodogram type:
%            'c' : classical periodogram.
%            's' : Scargle periodogram.
%            defualt is 'c' (classical).
%          - Display mesh at end of calculation.
%            'y' or 'n'. default is 'y'.
% Output : - Matrix of powers each column contain power for each
%            sub epoch.
%          - Matrix of leading frequency. Not Yet Avaliable.
%          - vector of sub epochs mid time.
%          - vector of frequencies searched.
% Remarks: Not all the option were checked. Please report
%          any bug/anomalies.
% Example: [P,F,T,Fr]=periodit(TimeSeries,[30,0.1],[],[],'s');
% Tested : Matlab 5.1
%     By : Eran O. Ofek             December 1997
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%----------------------------------------------------------------------
ColT = 1;
ColO = 2;
[Nobs,Ncol]=size(TimeSerMat);
if Ncol<2,
   error('Number of columns in the TimeSerMat should be at least 2');
end

% check number of arguments.
if nargin==1,
   EpochPars  = [];
   TaperPars  = [];
   FreqVec    = [];
   PeriodType = 'c';   % periodogram default is classical (periodia.m).
   Disp       = 'y';   % default. display mesh.
elseif nargin==2,
   TaperPars  = [];
   FreqVec    = [];
   PeriodType = 'c';   % periodogram default is classical (periodia.m).
   Disp       = 'y';   % default. display mesh.
elseif nargin==3,
   FreqVec    = [];
   PeriodType = 'c';   % periodogram default is classical (periodia.m).
   Disp       = 'y';   % default. display mesh.
elseif nargin==4,
   PeriodType = 'c';   % periodogram default is classical (periodia.m).
   Disp       = 'y';   % default. display mesh.
elseif nargin==5,
   Disp       = 'y';   % default. display mesh.
else
   error('Number of input arguments should be 1,2,3,4 or 5');
end

TimeSpan = TimeSerMat(Nobs,ColT) - TimeSerMat(1,ColT); 

% handling the Epoch Parameters vector (EpochPars).
[EpochParsSizeI,EpochParsSizeJ]=size(EpochPars);
if (EpochParsSizeI==0),
   % null EpochPars
   % use defaults
   EpochNum    = 10;    % number of epochs in time span.
   EpochLength = 0.2;   % epoch length in units of time span.
else
   if (length(EpochPars)~=2),
      error('Number of elements in EpochPars vector should be 2');
   end
   EpochNum    = EpochPars(1);
   EpochLength = EpochPars(2);
end

% handling the tapering function parameters (TaperPars).
[TaperParsSizeI,TaperParsSizeJ]=size(TaperPars);
if (TaperParsSizeI==0),
   % null TaperPars
   % use defaults
   TaperFun  = 'C';   % cosine bell tapering function.
   TaperFunP = 0.6;    % tapering function parameter.
                       % in the case of 'C' this is the precentage
                       % of the flat part.
else
   if (length(TaperPars)==1),
      if TaperPars(1)=='F',
         TaperFun = 'F';     % Flat (uniform) tapering function;
      else
         error('Unknown Tapering function');
      end
      % add others single parameters tapering function
   elseif (length(TaperPars)==2),
      if TaperPars(1)=='C',
         TaperFun = 'C';     % cosine bell tapering function;
      else
         error('Unknown Tapering function');
      end
      % add others single parameters tapering function
   else
      error('Number of elements in TaperPars vector should be null, 1 or 2');
   end
end

% handling the Frequncy Vector (FreqVec)
[FreqVecSizeI,FreqVEcSizeJ]= size(FreqVec);
if (FreqVecSizeI==0),
   % null FreqVec
   % use defaults
   LowFreq  = 0;
   HighFreq = 1./(2.*min(abs(diff(TimeSerMat(:,ColT)))));
   FreqInt  = 1./(2.*TimeSpan);
else
   if (length(FreqVec)~=3),
      error('Number of elements in FreqVec should be null or 3');
   else
      LowFreq  = FreqVec(1);
      HighFreq = FreqVec(2);
      FreqInt  = FreqVec(3);
   end
end


% start program
StartEpoch = TimeSerMat(1,ColT);
EpochLenT  = EpochLength.*TimeSpan;
EpochStep  = TimeSpan./EpochNum;

%check P size

if (PeriodType=='c'),
   [P,F] = periodia(TimeSerMat(1:ceil(0.1.*Nobs),:),LowFreq,HighFreq,FreqInt);
elseif (PeriodType=='s'),
   [P,F] = periodis(TimeSerMat(1:ceil(0.1.*Nobs),:),LowFreq,HighFreq,FreqInt);
else
   error('unknown periodogram type');
end
FreqSer    = P(:,1);
SuperP     = zeros(length(P(:,1)),EpochNum-1);
SuperF     = zeros(10,EpochNum-1);
MidTime    = zeros(1,EpochNum-1);
for Ind=0:1:EpochNum-2,
   JJ = find(TimeSerMat(:,ColT) >= (StartEpoch+Ind.*EpochStep) & TimeSerMat(:,ColT) < (StartEpoch+Ind.*EpochStep+EpochLenT));
   NewMat = TimeSerMat(JJ,:);
   %NewMat = rangebrk(TimeSerMat, StartEpoch+Ind.*EpochStep, StartEpoch+Ind.*EpochStep+EpochLenT,ColT);
   if (length(NewMat(1,:))==1),
      % NewMat is empty. continue.
      MidTime(1,Ind+1) = StartEpoch+Ind.*EpochStep + 0.5.*EpochLenT;
   else
      MidTime(1,Ind+1) = StartEpoch+Ind.*EpochStep + 0.5.*EpochLenT;
      % tapering
      [WeightFun,Range] = cosbell(TaperFunP,NewMat(:,ColT));
      NewMat(:,ColO) = NewMat(:,ColO).*WeightFun;
      if (PeriodType=='c'),
         [P,F] = periodia(NewMat,LowFreq,HighFreq,FreqInt);
      elseif (PeriodType=='s'),
         [P,F] = periodis(NewMat,LowFreq,HighFreq,FreqInt);
      else
         error('unknown periodogram type');
      end
      SuperP(:,Ind+1) = P(:,2);
      %SuperF(:,Ind+1) = F(length(F(:,1))-9:length(F(:,1)),1);
   end
end

if (Disp=='y'),
   mesh(MidTime,FreqSer,SuperP);
   xlabel('Time')
   ylabel('Frequency')
   zlabel('Power')
end

                 perioent.m                                                                                          0100644 0056337 0000144 00000004015 07713543350 011750  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function S=perioent(x,FreqVec,m);
%------------------------------------------------------------------------
% perioent function     Periodogram using information entropy.
%                     For each trail period, the phase-magnitude space
%                     is divided into m by m cells, and the information
%                     entropy is calculated.
%                     Then the probability to find observations
%                     in each square is MU(i) and the Entropy is
%                     (-Sigma{MU*ln(MU)}).
%                     The output Entropy is normalized by ln(m*m).
% input  : - Data matrix, sorted by time.
%          - Frequency Vector, [Lower Freq., Higher Freq., Freq. Interval]
%          - Square root of number of elements in the unit square. 
% output : - Two column matrix, [Frequency, Entropy].
% reference: Cincotta, Mendez & Nunez 1995, ApJ 449, 231-235.
% Tested : Matlab 5.1
%     By : Eran O. Ofek           November 1996
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%------------------------------------------------------------------------


c_x = 1;
c_y = 2;

NumObs = length(x(:,c_x));

LowFreq = FreqVec(1);
HighFreq = FreqVec(2);
FreqInt = FreqVec(3);

% find maximun and minimum of magbnitude for normalization
MinMag = min(x(:,c_y));
MaxMag = max(x(:,c_y));
DelMag = MaxMag - MinMag;

EntMat = zeros(m,m);
S = zeros(length([LowFreq:FreqInt:HighFreq]),2);

N = 1;
for Freq=LowFreq:FreqInt:HighFreq,
   temp1 = x(:,c_x).*Freq;
   phase = temp1 - floor(temp1);
   % calculating the density matrix
   x_pos = floor(phase.*m) + 1;
   y_pos = floor(m.*(x(:,c_y) - MinMag)./(DelMag+1e-5)) + 1;
   % calculating the entropy for the trail period
   for k=1:1:NumObs,
      EntMat(y_pos(k), x_pos(k)) = EntMat(y_pos(k), x_pos(k)) + 1;
   end
   % shold be equal to NumObs
   if sum(sum(EntMat))~=NumObs,
    error('ne');
   end
   mu = EntMat./NumObs;
   S(N,2) = -sum(sum(mu.*log(mu+eps)));
   S(N,1) = Freq;
   N = N + 1;
   EntMat = zeros(m,m);
end

% normalize the Entropy
S(:,2) = S(:,2)./log(m.*m);


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   phot_event_me.m                                                                                     0100644 0056337 0000144 00000004014 07713543651 012762  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [S,Sig]=phot_event_me(x,FreqVec,m);
%--------------------------------------------------------------------
% phot_event_me function     Searching periodicity in time-tagged
%                     events using information entropy.
%                     For each trail period, the phase-magnitude space
%                     is divided into m by m cells, and the information
%                     entropy is calculated.
%                     Then the probability to find observations
%                     in each square is MU(i) and the Entropy is
%                     (-Sigma{MU*ln(MU)}).
%                     The output Entropy is normalized by ln(m*m).
% input  : - Sorted vector of events.
%          - Frequency Vector, [Lower Freq., Higher Freq., Freq. Interval]
%          - number of elements in phase (number of bins). 
% output : - Entropy vs. Frequency.
%            This is a two column matrix. The first column contains
%            the frequency and the second list the entropy.
%          - One sigma probability.
% Reference : Cincotta et al. 1999, MNRAS 302,582.
% Tested : Matlab 5.2
%    By  Eran O. Ofek           April 1999
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------


NumObs = length(x);

LowFreq = FreqVec(1);
HighFreq = FreqVec(2);
FreqInt = FreqVec(3);


EntVec = zeros(m);
S = zeros(length([LowFreq:FreqInt:HighFreq]),2);

N = 1;
for Freq=LowFreq:FreqInt:HighFreq,
   phase = x.*Freq - floor(x.*Freq);
   % calculating the density vector
   x_pos = floor(phase.*m) + 1;
   % calculating the entropy for the trail period
   for k=1:1:NumObs,
      EntVec(x_pos(k)) = EntVec(x_pos(k)) + 1;
   end
   % shold be equal to NumObs
   if sum(sum(EntVec))~=NumObs,
    error('ne');
   end
   mu = EntVec./NumObs;
   S(N,2) = -sum(sum(mu.*log(mu+eps)));
   S(N,1) = Freq;
   N = N + 1;
   EntVec = zeros(m);
end

% normalize the Entropy
S(:,2) = S(:,2)./log(m);

% statistics - estimate one sigma
Sig = sqrt(1./NumObs).*((log(NumObs)+1)./log(m) - 1);


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    poisson_event.m                                                                                     0100644 0056337 0000144 00000003071 07713564610 013021  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [N,Nt,t]=poisson_event(Events,Bin_Size,Round);
%---------------------------------------------------------------------------
% poisson_event function     Given a vector of time-tagged events,
%                          compare the delta-time between successive events
%                          with the exponential distribution.
% Input  : - Vector of sorted time-tagged events
%          - Bin size, for calculating the delta-time histogram.
%          - Roundoff error, default = 0 sec.
% Output : - Observed interval per bin
%          - Calculated intervals per bin
%          - Vector of delta-time bins used.
% Tested : Matlab 5.2
%     By : Eran O. Ofek           Feb 1999
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%---------------------------------------------------------------------------

if (nargin<3),
   Round = 0;
end

Event_Int = sort(diff(Events));
MaxInt = max(Event_Int);
CellNo = ceil(MaxInt./Bin_Size);
UpperBound = CellNo.*Bin_Size;
[x,n] = realhist(Event_Int,CellNo,[0 UpperBound]);
% plot the observed distribution
bar(x,n);

hold on
% plot the calculated distribution
N  = length(Event_Int);
T  = mean(Event_Int);
Dt = Bin_Size;
t  = [0:Bin_Size:UpperBound-Bin_Size]';
if (Round==0),
   Nt = N.*Dt.*(exp(-t./T))./T;
else
   Nt = N.*T.*Dt.*((exp(Round./T) - 2 + exp(-Round./T)).*exp(-t.*Round./T))./Round;
end
h=plot(t+0.5.*Dt,Nt,'r');
set(h,'LineWidth',2);
xlabel('Time Interval');
ylabel('Number');
hold off;

% plot the number of sigma from exponential distribution from each bin
figure(2);
Sigma = (n - Nt)./sqrt(n);
plot(t+0.5.*Dt,Sigma);



                                                                                                                                                                                                                                                                                                                                                                                                                                                                       polysubs.m                                                                                          0100644 0056337 0000144 00000002133 07713545043 012003  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [y,a]=polysubs(x,n,c_x,c_y);
%-------------------------------------------------------------------------
% polysubs function       Subtract polynomial from a data set (no errors).
% Input  : - Data matrix.
%          - Degree of polynom to subtract.
%          - c_x, column of dependent variable, default is 1.
%          - c_y, column of independent variable, default is 2.
% Output : - Data matrix after the polynomial subtraction.
%            The other columns of original matrix are copied to y
%            only if c_x=1 and c_y=2.
%          - Polynomial coefecient of fit. 
% Tested : Matlab 4.2
%     By : Eran O. Ofek           May 1994
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%-------------------------------------------------------------------------
if nargin==2,
   c_x = 1
   c_y = 2;
elseif nargin==3,
   c_y = 2;
elseif nargin==4,
   % do nothing
else
   error('Illegal number of input arguments');
end

wid = length(x(1,:));
a = polyfit(x(:,c_x),x(:,c_y),n);
temp = [x(:,c_x), x(:,c_y) - polyval(a,x(:,c_x))];
if wid>c_y,
   y = [temp,x(:,c_y+1:wid)];
else
   y = temp;
end
                                                                                                                                                                                                                                                                                                                                                                                                                                     ps_batch.m                                                                                          0100644 0056337 0000144 00000001641 06774632410 011712  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function StarNum = ps_batch(Mat,MatInd,Thresh);
%---------------------------------------------------------
% ps_batch        generate power spectrum for all
%               stars and register these that have
%               peaks exceeding Thresh.
% Input  : - matrix of JD,Mag,Error
%          - matrix of indices, [StarNum, StD, FirstLine, LastLine]
%          - Threshold for power spectrum
% Output : - Numbers of stars exceeding Thresh
%     By Eran O. Ofek            Febuary 1999
%---------------------------------------------------------

C_t = 1;

List = 0;

for I=1:1:length(MatInd(:,1)),
   Low_Freq  = 0;
   High_Freq = 5;
   RangedData = Mat(MatInd(I,3):MatInd(I,4),:);
   Freq_Int  = 0.25./(max(RangedData(:,C_t)) - min(RangedData(:,C_t))) 
   [p,f]=periodia(RangedData,Low_Freq,High_Freq,Freq_Int);
   if (max(f(:,4))>Thresh),
      List = [List;MatInd(I,1)];
   end
end

N = length(List);
StarNum = List(2:N);



                                                                                               rankcorr.m                                                                                          0100644 0056337 0000144 00000001614 07646500125 011744  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [d,c]=rancorr(x,c1,c2);
%--------------------------------------------------------------------
% rankcorr function       Spearman's rank-correlation coefficient
% input  : - matrix
%          - J.D. column. defualt 1.
%          - Mag. column. defualt 2.
% output : - differences vector.
%          - Spearman's rank-correlation coefficient.
%    By  Eran O. Ofek           November 1993
%--------------------------------------------------------------------
if nargin==1,
   c1=1;
   c2=2;
elseif nargin==2,
   error('1 or 3 args only');
elseif nargin>3,
   error('1 or 3 args only');
end
len = length(x(:,1:1));
wid = length(x(1:1,:)) + 1;
temp = sortrows(x,c2,c1);
tem0 = 1:1.0:len;
tem0 = rot90(tem0,3);
temp(:,wid:wid) = tem0;
temp = sortrows(temp,c1);
d = abs(diff(temp(:,wid:wid)));
d(len) = abs(temp(len,wid:wid) - temp(1,wid:wid));
s = sum(d.*d);
con = 6;
c = 1 - con.*s/(len*len*len - len);
                                                                                                                    runderiv.m                                                                                          0100644 0056337 0000144 00000004113 07713545613 011764  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function RunDeriv=runderiv(Data,WindowWidth,TimeVec);
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
                                                                                                                                                                                                                                                                                                                                                                                                                                                     runmean.m                                                                                           0100644 0056337 0000144 00000007255 10260534426 011574  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function RunMean=runmean(Data,WFun,WFunPars,WS,TimeVec);
%-------------------------------------------------------------------------
% runmean function      Calculate the runing mean of an unevenly spaced
%                     time series with different weight functions and
%                     weight scheme.
% Input  : - Data matrix, [Time, Value, Error].
%            Error is optional, if not given then assumes
%            equal weights.
%          - Weight function type:
%            'f' : flat function:
%                  parameter is : i) total width
%            'g' : gaussian function (truncated at 3\sigma):
%                  parameter is : i) HWHM width
%            'c' : cosine-bell function:
%                  parameters are: i) total width
%                                 ii) fraction of flat part
%          - Weight function vector of parameters.
%          - Weighting scheme:
%            'f2' - use weight function squared only, default.
%            'f'  - use weight function only.
%            'wm' - use points-errors (weighted mean) only.
%            'wf' - use sum of squares of weight function and
%                   normalize points-errors so their mean is RelWeight.
%          - Time vector for which to calculate the running mean,
%            default is Data(:,1).
% Output : - Runing mean matrix [Time, Value].
% Example: runmean(Data,'c',[30 0.5],'wf');
% Tested : Matlab 5.3
%     By : Eran O. Ofek             January 2002
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%-------------------------------------------------------------------------
RelWeight = 0.1;    % relative weight between errors and weight function
TCol = 1;
YCol = 2;
ECol = 3;
if (nargin==4),
   TimeVec = Data(:,TCol);
elseif (nargin==5),
   % do nothing
else
   error('Illegal number of input arguments');
end

DataSpan = max(Data(:,1)) - min(Data(:,1));
SizeData = size(Data);
if (SizeData(2)==2),
   Data = [Data, ones(SizeData(1),1)];
end


% build weight function
switch WFun
 case 'f' 
    % flat function
    X     = [-0.5.*WFunPars(1):0.01.*WFunPars(1):0.5.*WFunPars(1)].';
    Y     = ones(size(X));
   
 case 'g'
    % truncated gaussian
    SC    = 3; % sigma cutoff  
    HWHM  = WFunPars(1);
    Sigma = sqrt(HWHM.^2./(2.*log(2)));
    X     = [-SC.*Sigma:0.02.*Sigma:SC.*Sigma].';
    Y     = exp(-X.^2./(2.*Sigma.^2));
    HalfR = SC.*Sigma;  % half range
    
 case 'c' 
    % cosine bell
    Range        = [-0.5.*WFunPars(1):0.01.*WFunPars(1):0.5.*WFunPars(1)].';
    [Y,X] = cosbell(WFunPars(2),Range);
    
 otherwise
    error('Unknown WFun type');
end

NormErr = RelWeight.*Data(:,ECol)./mean(Data(:,ECol));
InterpMethod = 'linear';
N  = length(TimeVec);
RM = zeros(N,1);
for I=1:1:N,
   if (isnan(TimeVec(I))==0),
      % points weight
      Wp = interp1(X+TimeVec(I),Y,Data(:,TCol),InterpMethod);
      
      switch WS
       case 'f2'
          % 'f2' - use weight function squared only, default.
          RM(I) = nansum(Wp.^2 .* Data(:,YCol))./nansum(Wp.^2);
          
       case 'f'
          % 'f'  - use weight function only.
          RM(I) = nansum(Wp .* Data(:,YCol))./nansum(Wp);
          
       case 'wm'
          % 'wm' - use points-errors (weighted mean) only.
          RM(I) = nansum(ceil(Wp) .* Data(:,YCol) .* (1./Data(:,ECol)).^2)./nansum(ceil(Wp) .* (1./Data(:,ECol)).^2);
          
       case 'wf'
          % 'wf' - use sum of squares of weight function and
          %        normalize points-errors so their mean is 1.
          RM(I) = nansum(Data(:,YCol) .* ((1./NormErr).^2 + Wp.^2))./nansum((1./NormErr).^2 + Wp.^2);
          
       otherwise
          error('Unknown Weights scheme type');
      end
   else
      RM(I) = NaN;
   end   
end

RunMean = [TimeVec, RM];
                                                                                                                                                                                                                                                                                                                                                   sf_interp.m                                                                                         0100644 0056337 0000144 00000004204 10316606500 012102  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function [InLC]=sf_interp(LC,BinSize,Time,InterpMethod,CCF);
%----------------------------------------------------------------------
% sf_interp function      Interpolation with structure function
%                       error propagation.
%                       The error bar in each interpolated point is
%                       calculated by adding in quadrature the
%                       the error of the neastest point with the
%                       amplitude of the stracture function at
%                       the the lag equal to the difference between
%                       the interpolated point and the nearest point.
% Input  : - Observations [Time, Mag, Err].
%          - Structure function binning interval.
%          - Vector of times for which to interpolate the LC.
%          - Interpolation method, default is 'linear'.
%          - Optional CCF matrix (see ccf.m) with the
%            structure function in columns [1 6 7].
%            If not given then calculate the structure function.
% Output : - Interpolated LC, [Time, Mag, Err].
% Tested : Matlab 5.3
%     By : Eran O. Ofek        December 2002
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.htm
% Reference : Ofek & Maoz 2003
%----------------------------------------------------------------------
ColT = 1;
ColM = 2;
ColE = 3;

CalcCCF = 'n';

if (nargin==3),
   InterpMethod = 'linear';
   CalcCCF      = 'y';
elseif (nargin==4),
   CalcCCF      = 'y';
elseif (nargin==5),
   CalcCCF      = 'n';
else
   error('Illegal number of input arguments');
end


switch CalcCCF
 case 'y'
    CCF  = ccf(LC,LC,BinSize,'normal');
    SF   = CCF(:,[1 6 7]);
 case 'n'
    % allready given
    SF   = CCF(:,[1 6 7]);
 otherwise
    error('Unknown CalcCCF option');
end

Ig = find(isnan(SF(:,2))==0);
SF = SF(Ig,:);
Ig = find(SF(:,2)<0);
SF(Ig,2) = 0;

Mag = interp1(LC(:,ColT),LC(:,ColM),Time);

Nt   = length(Time);
InLC = zeros(Nt,3);
InLC(:,1) = Time;
InLC(:,2) = Mag;
for I=1:1:Nt,
   DeltaT     = Time(I) - LC(:,ColT);
   [Min,MinI] = min(abs(DeltaT));
   %--- interpolate SF ---

   Amp2  = interp1(SF(:,1),SF(:,2),Min);
   Err   = sqrt(LC(MinI,ColE).^2 + Amp2);

   InLC(I,3) = Err;
end



                                                                                                                                                                                                                                                                                                                                                                                            specwin.m                                                                                           0100644 0056337 0000144 00000001702 07713547013 011573  0                                                                                                    ustar   eran                            users                                                                                                                                                                                                                  function Pxw = specwin(x,l_f,h_f,df,c_x);
%--------------------------------------------------------------------
% specwin function     spectral window of a time series.
% Input  : - Time series matrix.
%          - h_l, is the low frequency.
%          - h_f, is the high frequency.
%          - df, is the frequency interval.
%          - Time column, default is 1.
% Output : - Spectral window, [Freq, Power].
% Tested : Matlab 4.2
%     By : Eran O. Ofek           October 1994
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
c_y = 2;
if nargin==4,
   c_x = 1;
elseif nargin==5,
   % do nothing
else
   error('Illegal number of input arguments');
end

N0 = length(x(:,c_x));
f_ind = l_f;
k = 1;
Pxw = zeros((h_f-l_f)./df,2);
while f_ind<h_f,
   Pxw(k,1) = f_ind; 
   temp = abs(sum(exp(-i*2*pi*f_ind*x(:,c_x))));
   Pxw(k,2) = temp*temp/N0;
   f_ind = f_ind + df;
   k = k + 1;
end
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              