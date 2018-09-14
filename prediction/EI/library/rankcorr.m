function [d,c]=rancorr(x,c1,c2);
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
