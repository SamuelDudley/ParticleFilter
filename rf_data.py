import os
import numpy as np
import matplotlib.pyplot as plt

def get_rf_data():
    Headder = []
    
    path_to_data = "/home/uas/Desktop/woomera/compolated/pol_soo_test1"
    #path_to_data = "/home/uas/Desktop/woomera/compolated/soo_oct_wtr"

    rf_file = "rf_ap_data7.hawk"
    rf_data_file_path = os.path.join(path_to_data, rf_file)
    
    print rf_data_file_path
    num_lines = sum(1 for line in open(rf_data_file_path, 'r'))
    
    with open(rf_data_file_path, 'r') as fid:
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
                try:
                    Data[line_number-1,:] = line_data
                except:
                    print "!"
    
    return Data
"""            
print Headder
print Headder[6]
freq = 1360e6

text_offset = 3
a = Headder[text_offset:]
(index,value)= min(enumerate(a), key=lambda x: abs(float(x[1])-freq))
index += text_offset
print index, value
"""
if __name__ == '__main__':
    Data = get_rf_data()
    xlim_lower = 100
    xlim_upper = 3000
    x = Data[:,1]
    y1 = Data[:,6]#there is data at 6 (~1280), 59 (~1320) and 112 (~1360)
    y2 = Data[:,59]
    y3 = Data[:,112]
    plt.subplot(3, 1, 1)
    plt.plot(x,y1)
    plt.xlim(xlim_lower,xlim_upper)
    plt.subplot(3, 1, 2)
    plt.plot(x,y2)
    plt.xlim(xlim_lower,xlim_upper)
    plt.subplot(3, 1, 3)
    plt.plot(x,y3)
    plt.xlim(xlim_lower,xlim_upper)
    plt.show()

#400 to 1200