function y=folding(x,p,c);
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
