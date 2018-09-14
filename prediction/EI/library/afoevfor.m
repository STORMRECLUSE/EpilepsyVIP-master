function y=afoevfor2(file);
%--------------------------------------------------------------------
% afoevfor function    formating AFOEV observations into
%                    matlab variable.
% Input  : - file name, contains the variable star observation
%            in the AFOEV format.
% Output : - matrix of observations, in which the first column
%            contains the MJD and the second column contains the
%            visual magnitude.
% Tested : Matlab 5.0
%     By : Eran O. Ofek           July 1995
%  modified by Orly Gnat        August 1997
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%--------------------------------------------------------------------
fid = fopen(file,'r');
line = 0;
while line~=-1,
   line = fgetl(fid);
   if line==-1,
      break;
   end
   if line(12:13)==' ',
      jd = str2num(line(4:11));
      mg = str2num(line(14:16));
      y  = [y;[jd,mg]];
   end
end
fclose(fid);

