'''
Read from all scattering coefficient files
'''
from __future__ import print_function
import os
import csv
csv_file_dir = 'Users/vsp/Google Drive/Matlab/Scattering Coeffs/'
all_potential_files = os.listdir(csv_file_dir)
for filename in all_potential_files:
    if filename.startswith('labeled_'):
        pass