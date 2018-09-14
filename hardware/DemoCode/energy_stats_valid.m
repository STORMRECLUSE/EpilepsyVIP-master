function compare_energy_stats(sine_file,energy_file,window_size,window_step)
n_plot = 4;
S_data = load(sine_file);
patient_data = S_data.record_RMpt5;
n_channels = size(patient_data,1);
chann_len = size(patient_data,2);

[matlab_coeffs,num_win]= compute_gardner_coeffs(patient_data,window_size,window_step);

c_coeffs = read_gardner_coeffs(energy_file,n_channels,num_win);
plot_coeffs(matlab_coeffs,c_coeffs(:,:,1:end-1),5)
return

function [matlab_coeffs,num_win]= compute_gardner_coeffs(patient_data,window_size,window_step)


n_channels = size(patient_data,1);
chann_len = size(patient_data,2);
num_win = 1 + floor((chann_len - window_size)/window_step);

matlab_coeffs= zeros(3,num_win,n_channels);
win_index = 1;
for window_start = 1:window_step:(chann_len-window_size)
window_indices = window_start:(window_start + window_size -1);
matlab_coeffs(1,win_index,:) = ...
log(1/window_size*sum(abs(diff(patient_data(:,window_indices),1,2)),2));

matlab_coeffs(2,win_index,:) = ...
log(1/window_size*sum(patient_data(:,window_indices(2:end)).^2,2));

matlab_coeffs(3,win_index,:) = ...
log(1/window_size*sum(abs(patient_data(:,window_indices(2:end-1)).^2-...
        patient_data(:,window_indices(1:end-2)).*...
        patient_data(:,window_indices(3:end),:)),2));

win_index = win_index + 1;
end


return

function c_coeffs = read_gardner_coeffs(energy_file,n_channels,n_win)
e_hand = fopen(energy_file);
energy_data = fscanf(e_hand,'%f ');
c_coeffs = reshape(energy_data,3,n_win,155);

%c_coeffs = permute(c_coeffs,[1 3 2]);

%resha


return

function plot_coeffs(matlab_coeffs,c_coeffs, n_plot)
for channel = 1:n_plot
subplot(1,n_plot,channel)
plot(squeeze(matlab_coeffs(:,:,channel))','-');
hold on

plot(squeeze(c_coeffs(:,:,channel))','--');
xlabel('Sample Number')
ylabel('Feature Quantity')

end
return