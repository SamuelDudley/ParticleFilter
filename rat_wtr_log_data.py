import numpy as np
import Model
import matplotlib.pyplot as plt
from math import *

def haversine(lon1, lat1, lon2, lat2):
    
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2.)**2 + cos(lat1) * cos(lat2) * sin(dlon/2.)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371*1000. # Radius of earth in meters. 
    return c * r

def find_nearest(array,value):
    #note: assumes sorted array! okay for time based arrays...
    idx = np.searchsorted(array, value, side="left")
    if idx >= len(array):
        return len(array)-1#array[-1]
    if math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
        return idx-1#array[idx-1]
    else:
        return idx#array[idx]

y_rf = np.load('RSSI.npy')
x_rf = np.load('TimeRAD.npy')

lat = np.load('Lat.npy')
lng = np.load('Lng.npy')
alt = np.load('Alt.npy')
x_ap = np.load('TimeAHR.npy')

alt = alt-130.0
y_rf = y_rf -180
y_rf = y_rf
gcs_tx = np.vectorize(haversine)(lng, lat, 136.54463,-30.93484)

gcs = {'rf':0, 'ap':-1, 'freq':918}
transmitter = gcs
tx_height = [12.0]

two_ray = np.vectorize(Model.twoRay)(gcs_tx, transmitter['freq'], tx_height[0], alt, 200000)
# 
# # free_space = np.vectorize(Model.freeSpace)(ap[:,transmitter['ap']], transmitter['freq'])
# 
# for idx, sample_time in enumerate(x_rf):
#     #find the closest time to the true sample time in the model list (based on ap data)
#     near_idx_ap = find_nearest(x_ap, sample_time)
#     near_idx_rf = find_nearest(x_rf, sample_time)
#     print sample_time, x_ap[near_idx_ap], x_rf[near_idx_rf]
#     print y_rf1[near_idx_rf], two_ray1[near_idx_ap], free_space[near_idx_ap]
#     error_2ray[idx] =  (dbm_to_pwr(y_rf[near_idx_rf]) - dbm_to_pwr(two_ray1[near_idx_ap]))**2
#     error_fsl[idx] = (dbm_to_pwr(y_rf[near_idx_rf]) - dbm_to_pwr(free_space[near_idx_ap]))**2
#     error_2ray[idx] =  (abs(y_rf[near_idx_rf]) - abs(two_ray1[near_idx_ap]))**2
#     error_fsl[idx] = (abs(y_rf[near_idx_rf]) - abs(free_space[near_idx_ap]))**2


plt.plot(x_rf,y_rf)
plt.plot(x_ap, two_ray)
plt.show()