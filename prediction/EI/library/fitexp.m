function [NewPar,NewParErr,Chi2,Deg,Cov]=fitexp(X,Y,DelY);
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

