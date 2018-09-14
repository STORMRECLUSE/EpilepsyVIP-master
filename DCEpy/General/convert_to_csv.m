function convert_to_csv(file_name)
% Converts an edf file to a csv file
% Relies on edfread, which is code we already had
% Input:
%   Edf file name, do not include extension
% Ouput:
%   Saves it as same file name, just .csv

[h, r] = edfread([file_name '.edf'], 'assignToVariables', true);

all_data = [];

for i = 1:length(h.label)
    lbl = h.label{i};
    all_data = [all_data (eval(lbl)')];    
end

fid = fopen([file_name '_labels1.csv'],'wt');
[rows,cols]=size(h.label);
for i = 1:rows
      fprintf(fid,'%s,',h.label{i,1:end-1})
      fprintf(fid,'%s\n',h.label{i,end})
end
fclose(fid);

size(all_data);

csvwrite([file_name '.csv'], all_data,1,0);

end