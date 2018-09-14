function StarNum = ps_batch(Mat,MatInd,Thresh);
%---------------------------------------------------------
% ps_batch        generate power spectrum for all
%               stars and register these that have
%               peaks exceeding Thresh.
% Input  : - matrix of JD,Mag,Error
%          - matrix of indices, [StarNum, StD, FirstLine, LastLine]
%          - Threshold for power spectrum
% Output : - Numbers of stars exceeding Thresh
%     By Eran O. Ofek            Febuary 1999
%---------------------------------------------------------

C_t = 1;

List = 0;

for I=1:1:length(MatInd(:,1)),
   Low_Freq  = 0;
   High_Freq = 5;
   RangedData = Mat(MatInd(I,3):MatInd(I,4),:);
   Freq_Int  = 0.25./(max(RangedData(:,C_t)) - min(RangedData(:,C_t))) 
   [p,f]=periodia(RangedData,Low_Freq,High_Freq,Freq_Int);
   if (max(f(:,4))>Thresh),
      List = [List;MatInd(I,1)];
   end
end

N = length(List);
StarNum = List(2:N);



