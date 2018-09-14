function [Par,Chi2,Freedom,Par1,Resid]=fitharmonw(X,Y,Har,Deg);
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
