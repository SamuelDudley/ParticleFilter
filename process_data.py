#this script will process WTR data

import os
import numpy as np
import matplotlib.pyplot as plt
import rf_data
import ap_telem_data
import Model
import math

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

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

rf = rf_data.get_rf_data()
ap = ap_telem_data.get_telem_data()
time_offset = min([min(rf[:,0]), min(ap[:,0])])
print time_offset
rf[:,0] = np.vectorize(subtract)(rf[:,0],time_offset)
ap[:,0] = np.vectorize(subtract)(ap[:,0],time_offset)
x_rf = rf[:,0]
x_ap = ap[:,0]
alt_ap = ap[:,4]
alt_ap = np.vectorize(subtract)(alt_ap,321.25)


#NORTH, WEST, SOUTH
north = {'rf':6, 'ap':-3, 'freq':1280, 'height':3.2}
west = {'rf':59, 'ap':-2, 'freq':1320, 'height':3.2}
south = {'rf':112, 'ap':-1, 'freq':1360, 'height':3.3}


#530 to 1580s at ~100m AGL
#1580 to 3000s at ~200m AGL
limits = True
xlim_lower = 1750
xlim_upper = 1850

#tx height seems to fit into the 3.0 to 4.1 range
transmitter = north
y_rf1 = rf[:,transmitter['rf']]  #6,59,112
y_ap1 = ap[:,transmitter['ap']] #-3,-2,-1 
two_ray1 = np.vectorize(Model.twoRay)(ap[:,transmitter['ap']], transmitter['freq'], transmitter['height'], alt_ap, 50)

free_space = np.vectorize(Model.freeSpace)(ap[:,transmitter['ap']], transmitter['freq'])
free_space = np.vectorize(subtract)(free_space,-2.2041)

transmitter = west
y_rf2 = rf[:,transmitter['rf']]  #6,59,112
y_ap2 = ap[:,transmitter['ap']] #-3,-2,-1 
two_ray2 = np.vectorize(Model.twoRay)(ap[:,transmitter['ap']], transmitter['freq'], transmitter['height'], alt_ap, 70)

transmitter = south
y_rf3 = rf[:,transmitter['rf']]  #6,59,112
y_ap3 = ap[:,transmitter['ap']] #-3,-2,-1 
two_ray3 = np.vectorize(Model.twoRay)(ap[:,transmitter['ap']], transmitter['freq'], transmitter['height'], alt_ap, 70)

#dowel lengths, 1.8, 2.4, 3m
window = 3
MA_y_rf = movingaverage(y_rf1, window)
MA_y_2r = movingaverage(two_ray1, window)
MA_y_fs = movingaverage(free_space, window)


#need to interpolate the model to match the sample points.
start_time = 1150
end_time = 1400

start_idx = find_nearest(x_rf, start_time)
end_idx = find_nearest(x_rf, end_time)
print start_idx, end_idx

x_rf_trimmed = x_rf[start_idx:end_idx]
print x_rf_trimmed

error_2ray = np.zeros(len(x_rf_trimmed))
error_fsl = np.zeros(len(x_rf_trimmed))
for idx, sample_time in enumerate(x_rf_trimmed):
    #find the closest time to the true sample time in the model list (based on ap data)
    near_idx_ap = find_nearest(x_ap, sample_time)
    near_idx_rf = find_nearest(x_rf, sample_time)
#     print sample_time, x_ap[near_idx_ap], x_rf[near_idx_rf]
#     print y_rf1[near_idx_rf], two_ray1[near_idx_ap], free_space[near_idx_ap]
#     error_2ray[idx] =  (abs(y_rf1[near_idx_rf]) - abs(two_ray1[near_idx_ap]))**2
#     error_fsl[idx] = (abs(y_rf1[near_idx_rf]) - abs(free_space[near_idx_ap]))**2
    
    error_2ray[idx] =  (abs(MA_y_rf[near_idx_rf]) - abs(MA_y_2r[near_idx_ap]))**2
    error_fsl[idx] = (abs(MA_y_rf[near_idx_rf]) - abs(MA_y_fs[near_idx_ap]))**2
    
    
print "two ray error: ", np.mean(error_2ray)
print "free space loss error: ", np.mean(error_fsl)



#
plt.subplot(2, 1, 1)
# plt.plot(x_rf,y_rf1)
# plt.plot(x_ap, two_ray1)
plt.plot(x_rf[window-1:], MA_y_rf)
plt.plot(x_ap[window-1:], MA_y_2r)
# plt.plot(x_rf,y_rf2)
# plt.plot(x_ap, two_ray2)
# plt.plot(x_rf,y_rf3)
# plt.plot(x_ap, two_ray3)
# plt.plot(x_ap, free_space)
plt.plot(x_ap[window-1:], MA_y_fs)
if limits:
    plt.xlim(xlim_lower,xlim_upper)

plt.subplot(2, 1, 2)
plt.plot(x_ap,y_ap1)#plt.plot(x_ap,alt_ap)#
plt.plot(x_ap,y_ap2)#plt.plot(x_ap,alt_ap)#
plt.plot(x_ap,y_ap3)#plt.plot(x_ap,alt_ap)#

if limits:
    plt.xlim(xlim_lower,xlim_upper)

plt.show()
"""
if __name__ == '__main__':
"""