function [N,Nt,t]=poisson_event(Events,Bin_Size,Round);
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



