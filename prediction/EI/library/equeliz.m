function [LC1e,LC2e]=equeliz(LC1,LC2,Thresh);
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
