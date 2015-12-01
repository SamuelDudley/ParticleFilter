#this script will process WTR data

import os
import numpy as np
import matplotlib.pyplot as plt
import ast
from math import radians, cos, sin, asin, sqrt


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

def get_telem_data():
    Headder = []
    
#     path_to_data = "/home/uas/Desktop/woomera/compolated/pol_soo_test1"
    path_to_data = "/home/uas/Desktop/woomera/compolated/soo_oct_wtr"
#     path_to_data = "/home/uas/Desktop/woomera/cland2/flight_comp"

    ap_file = "ap_data8.hawk"
    ap_data_file_path = os.path.join(path_to_data, ap_file)
    count =0
    print ap_data_file_path
    num_lines = sum(1 for line in open(ap_data_file_path, 'r'))
    print num_lines
    with open(ap_data_file_path, 'r') as fid:
        for (line_number, line_data) in enumerate(fid):
            if line_number == 0:
                line_data = line_data.split(' ', 2)
                line_data[0] = line_data[0].lstrip('#')
                line_data[-1] = line_data[-1].rstrip('\n')
                
                Headder = line_data
                print Headder
                
                #Data = np.zeros((num_lines-2, len(Headder)))
                Data = np.zeros((1,8),np.float64)
                #the -2 takes into account the headder line and removal of
                #the last line (which is incomplete)
                #(rows,cols)
                
            else:
                line_data = line_data.split(' ',2)
                line_data[-1] = line_data[-1].rstrip('\n')
                #convert to float
                try:
                    dict = ast.literal_eval(line_data[-1])
                    if dict['mavpackettype']=='GLOBAL_POSITION_INT':
                        #print line_data[0],line_data[1],  dict['lat'], dict['lon'], dict['hdg']
                        pass
                    if dict['mavpackettype']=='GPS_RAW_INT':
                        #print line_data[0],line_data[1],  dict['lat'], dict['lon'], dict['alt'], dict['relative_alt']
                        pass
                except:
                    pass
                    #print '!',line_data[-1]
                    
                if dict['mavpackettype']=='AHRS2':
                    #print line_data[0],line_data[1],  dict['lat'], dict['lng'], dict['altitude'], dict['roll'], dict['pitch'], dict['yaw']
                    
                    Data = np.append(Data, [[line_data[0],line_data[1], dict['lng']/1e7, dict['lat']/1e7, dict['altitude'], dict['roll'], dict['pitch'], dict['yaw']]], axis=0)
                for data_idx, data_val in enumerate(line_data):
                    try:
                        data_val = float(data_val)
                    except:
                        pass
                        #print (data_idx, data_val)
                """
                try:
                    Data[line_number-1,:] = line_data
                except:
                    count +=1
                """
    Data = Data.astype(dtype=np.float64)
    Data = Data[1:,:]
    
    northern_tx = np.vectorize(haversine)(Data[:,2], Data[:,3],136.54456079, -30.9233928817)
    western_tx = np.vectorize(haversine)(Data[:,2], Data[:,3], 136.5386026, -30.9277435856)
    southern_tx = np.vectorize(haversine)(Data[:,2], Data[:,3], 136.5463575,-30.93537259)
    
    northern_tx = northern_tx.reshape((len(northern_tx), 1))
    western_tx = western_tx.reshape((len(western_tx), 1))
    southern_tx = southern_tx.reshape((len(southern_tx), 1))
    
    Data = np.append(Data, northern_tx, axis=1)
    Data = np.append(Data, western_tx, axis=1)
    Data = np.append(Data, southern_tx, axis=1)
    return Data

if __name__ == '__main__':

    Data = get_telem_data()
    x = Data[:,2]
    y = Data[:,3]
    plt.plot(x,y)
    plt.scatter(136.544507, -30.925671)
    plt.scatter(136.538676, -30.927742)
    plt.scatter(136.546741,-30.9349)
    
    plt.show()

    x=Data[:,1]
    y1=Data[:,-1]
    y2=Data[:,-2]
    y3=Data[:,-3]
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.show()
    #400 to 1200