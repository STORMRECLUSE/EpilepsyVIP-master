function [OutJD,ObjVel]=hjd(JD,ObjCoo,Type);
%---------------------------------------------------------------------------
% hjd function         Convert Julian Day (UTC) to Helicentric/Barycentric
%                    Julian Day (for geocentric observer).
% Input  : - Column vector of julian days (UTC time system).
%          - J2000.0 object coordinates, [RA, Dec], in radians.
%          - Observer Geodetic position,
%            [East_Long (rad), North_Lat (rad), Geodetic height (meters)].
%            If geodetic position is not given (or empty matrix),
%            then assume geocentric observer.
%          - Output type:
%            'lh' - low accuracy heliocentric JD.
%            'hh' - high accuracy (full VSOP87) heliocentric JD.
%            'hb' - high accuracy (full VSOP87) barycentric JD, default.
% Output : - Heliocentric/Barycentric julian day (for geocentric observer).
%          - Heliocentric/Barycentric velocity comonent in object
%            direction [km/sec] (only for 'hh' | 'hb' options).
% Example : [OutJD,ObjVel]=hjd(JD,[RA Dec],'hb');
% Tested : Matlab 5.3 
%     By : Eran O. Ofek                  November 1993
%                          Last update : August 2003
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%---------------------------------------------------------------------------
  SEC_IN_DAY = 86400.0;
if (nargin==2),
   Type = 'hb';
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end

C  = get_constant('c','agd');
AU = get_constant('au','SI')./1000;
N  = length(JD);

switch Type
 case 'lh'
    [RA,Dec]     = suncoo(JD,'j');
    Coo          = cosined([RA+pi, -Dec]).';
    Vel          = [NaN; NaN; NaN];
    Vel          = repmat(Vel,1,N);
 case 'hh'
    [DeltaT,DUT] = delta_t(JD);
    JD_TT        = JD + (DUT + DeltaT)./SEC_IN_DAY; 
    [Coo,Vel]    = calc_vsop87(JD_TT, 'Earth', 'a', 'E');

 case 'hb'
    [DeltaT,DUT] = delta_t(JD);
    JD_TT        = JD + (DUT + DeltaT)./SEC_IN_DAY; 
    [Coo,Vel]    = calc_vsop87(JD_TT, 'Earth', 'e', 'E');

 otherwise
    error('Unknown Type option');
end

ObjPos = [cosined(ObjCoo)].';


DelJD = zeros(N,1);
ObjVel = zeros(N,1);
for I=1:1:N,
   EarthCoo = Coo(:,I);
   EarthVel = Vel(:,I);

   DelJD(I)  = norm(EarthCoo).*dot(ObjPos./norm(ObjPos),EarthCoo./norm(EarthCoo))./C;
   ObjVel(I) = norm(EarthVel).*dot(ObjPos./norm(ObjPos),EarthVel./norm(EarthVel)).*AU./SEC_IN_DAY;
end

OutJD = JD + DelJD;

DelJD*86400
%ObjVel


