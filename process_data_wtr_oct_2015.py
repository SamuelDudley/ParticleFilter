#this script will process WTR data

import os
import numpy as np
import matplotlib.pyplot as plt
import uHard_data
import ap_telem_data_wtr
import Model
import math
import cell_data




def subtract(a,b):
    return a-b

def find_nearest(array,value):
    #note: assumes sorted array! okay for time based arrays...
    idx = np.searchsorted(array, value, side="left")
    if idx >= len(array):
        return len(array)-1#array[-1]
    if math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
        return idx-1#array[idx-1]
    else:
        return idx#array[idx]

def dbm_to_pwr(db):
    pwr = 1. * 10.**(db/10.) / 1000.
    return pwr

#cell = cell_data.get_rf_data()
rf = uHard_data.get_rf_data()
ap = ap_telem_data_wtr.get_telem_data()



x_rf = rf[:,0]
y_rf = rf[:,4]

x_ap = ap[:,0]

i = np.where(x_rf==0)
for a in i:
    x_rf = np.delete(x_rf, a, 0)
    y_rf = np.delete(y_rf, a, 0)
    
        
for idx, val in enumerate(x_rf):
    if val == 0:
        print 'zero'

time_offset = min([min(x_rf), min(x_ap)])
print time_offset
x_rf = np.vectorize(subtract)(x_rf,time_offset)
x_ap = np.vectorize(subtract)(x_ap,time_offset)

# x_cell = cell[:,0]
# y_cell = cell[:,12]
alt_ap = ap[:,4]
#alt_ap = np.vectorize(subtract)(alt_ap,321.25)
alt_ap = np.vectorize(subtract)(alt_ap,130.2)


#NORTH, WEST, SOUTH
north = {'rf':6, 'ap':-3, 'freq':1280}
west = {'rf':59, 'ap':-2, 'freq':1320}
south = {'rf':112, 'ap':-1, 'freq':1360}
gcs = {'rf':0, 'ap':-1, 'freq':1360}

transmitter = gcs
#530 to 1580s at ~100m AGL
#1580 to 3000s at ~200m AGL
limits = True
xlim_lower = 1000
xlim_upper = 1990
tx_height = [15., 11.8]#11.8#14.8
#tx height seems to fit into the 3.0 to 4.1 range

  #6,59,112
y_ap = ap[:,-1] #-3,-2,-1 

# #dowel lengths, 1.8, 2.4, 3m
two_ray1 = np.vectorize(Model.twoRay)(ap[:,transmitter['ap']], transmitter['freq'], tx_height[0], alt_ap, 200000)
two_ray2 = np.vectorize(Model.twoRay)(ap[:,transmitter['ap']], transmitter['freq'], tx_height[1], alt_ap, 200000)

free_space = np.vectorize(Model.freeSpace)(ap[:,transmitter['ap']], transmitter['freq'])

free_space = np.vectorize(subtract)(free_space,-39.8)

start_time = xlim_lower
end_time = xlim_upper

start_idx = find_nearest(x_rf, start_time)
end_idx = find_nearest(x_rf, end_time)
print start_idx, end_idx

x_rf_trimmed = x_rf[start_idx:end_idx]
print x_rf_trimmed

error_2ray = np.zeros(len(x_rf_trimmed))
error_fsl = np.zeros(len(x_rf_trimmed))

error_2ray_x = np.zeros(len(x_rf_trimmed))
error_fsl_x = np.zeros(len(x_rf_trimmed))

for idx, sample_time in enumerate(x_rf_trimmed):
    #find the closest time to the true sample time in the model list (based on ap data)
    near_idx_ap = find_nearest(x_ap, sample_time)
    near_idx_rf = find_nearest(x_rf, sample_time)
#     print sample_time, x_ap[near_idx_ap], x_rf[near_idx_rf]
#     print y_rf1[near_idx_rf], two_ray1[near_idx_ap], free_space[near_idx_ap]
    error_2ray[idx] =  (dbm_to_pwr(y_rf[near_idx_rf]) - dbm_to_pwr(two_ray1[near_idx_ap]))**2
    error_fsl[idx] = (dbm_to_pwr(y_rf[near_idx_rf]) - dbm_to_pwr(free_space[near_idx_ap]))**2
#     error_2ray[idx] =  (abs(y_rf[near_idx_rf]) - abs(two_ray1[near_idx_ap]))**2
#     error_fsl[idx] = (abs(y_rf[near_idx_rf]) - abs(free_space[near_idx_ap]))**2
    
    error_2ray_x[idx] =  x_ap[near_idx_ap]
    error_fsl_x[idx] = x_ap[near_idx_ap]
    
print "two ray error: ", np.mean(error_2ray)
print "free space loss error: ", np.mean(error_fsl)

#
plt.subplot(2, 1, 1)
plt.plot(x_rf,y_rf)
plt.plot(x_ap, two_ray1)
# plt.plot(x_ap, two_ray2)
plt.plot(x_ap, free_space)
# plt.plot(x_cell, y_cell)
if limits:
    plt.xlim(xlim_lower,xlim_upper)
    plt.ylim(-90,-30)

plt.subplot(2, 1, 2)
plt.plot(error_2ray_x, error_2ray)
plt.plot(error_fsl_x, error_fsl)

# plt.plot(x_ap,y_ap)
# plt.plot(x_ap,alt_ap)#
# if limits:
#     plt.xlim(xlim_lower,xlim_upper)

plt.show()
"""
if __name__ == '__main__':
"""