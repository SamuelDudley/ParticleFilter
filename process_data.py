#this script will process WTR data

import os
import numpy as np
import matplotlib.pyplot as plt
import rf_data
import ap_telem_data
import Model

def subtract(a,b):
    return a-b

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
north = {'rf':6, 'ap':-3, 'freq':1280}
west = {'rf':59, 'ap':-2, 'freq':1320}
south = {'rf':112, 'ap':-1, 'freq':1360}

transmitter = north
#530 to 1580s at ~100m AGL
#1580 to 3000s at ~200m AGL
limits = True
xlim_lower = 1150
xlim_upper = 1400
tx_height = 3.2
#tx height seems to fit into the 3.0 to 4.1 range

y_rf = rf[:,transmitter['rf']]  #6,59,112
y_ap = ap[:,transmitter['ap']] #-3,-2,-1 

#dowel lengths, 1.8, 2.4, 3m
two_ray = np.vectorize(Model.twoRay)(ap[:,transmitter['ap']], transmitter['freq'], tx_height, alt_ap, 40)
free_space = np.vectorize(Model.freeSpace)(ap[:,transmitter['ap']], transmitter['freq'])

 
#
plt.subplot(2, 1, 1)
plt.plot(x_rf,y_rf)
plt.plot(x_ap, two_ray)
plt.plot(x_ap, free_space)
if limits:
    plt.xlim(xlim_lower,xlim_upper)

plt.subplot(2, 1, 2)
plt.plot(x_ap,y_ap)#plt.plot(x_ap,alt_ap)#
if limits:
    plt.xlim(xlim_lower,xlim_upper)

plt.show()
"""
if __name__ == '__main__':
"""