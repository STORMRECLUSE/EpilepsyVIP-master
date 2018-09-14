function l=curvlen(x,c_x,c_y);
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
