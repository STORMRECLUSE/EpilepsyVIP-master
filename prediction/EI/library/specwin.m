function Pxw = specwin(x,l_f,h_f,df,c_x);
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
