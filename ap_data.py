#this script will process WTR data

import os
import numpy as np
import matplotlib.pyplot as plt

Headder = []

path_to_data = "/home/uas/Desktop/woomera/compolated/pol_soo_test1"
ap_file = "flight.hawk"
ap_data_file_path = os.path.join(path_to_data, ap_file)
count =0
print ap_data_file_path
num_lines = sum(1 for line in open(ap_data_file_path, 'r'))
print num_lines
with open(ap_data_file_path, 'r') as fid:
    for (line_number, line_data) in enumerate(fid):
        if line_number == 0:
            line_data = line_data.split(' ')
            line_data[0] = line_data[0].lstrip('#')
            line_data[-1] = line_data[-1].rstrip('\n')
            
            Headder = line_data
            
            Data = np.zeros((num_lines-2, len(Headder)))
            #the -2 takes into account the headder line and removal of
            #the last line (which is incomplete)
            #(rows,cols)
            
        else:
            line_data = line_data.split(' ')
            line_data[-1] = line_data[-1].rstrip('\n')
            #convert to float
            for data_idx, data_val in enumerate(line_data):
                try:
                    data_val = float(data_val)
                except:
                    print (data_idx, data_val)
            """
            try:
                Data[line_number-1,:] = line_data
            except:
                count +=1
            """

print count
#400 to 1200