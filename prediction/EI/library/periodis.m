function [Pxw, Fm] = periodis(x,l_f,h_f,df,pr);
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
