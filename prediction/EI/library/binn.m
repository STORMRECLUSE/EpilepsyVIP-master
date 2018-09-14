function y=binn(x,n);
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



