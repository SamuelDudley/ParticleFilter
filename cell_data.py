import os
import numpy as np
import matplotlib.pyplot as plt

def get_rf_data():
    Headder = []
    
    path_to_data = "/home/uas/Desktop/woomera/cland2/flight_comp"

    rf_file = "modem_status1.hawk"
    rf_data_file_path = os.path.join(path_to_data, rf_file)
    
    print rf_data_file_path
    num_lines = sum(1 for line in open(rf_data_file_path, 'r'))
    l1 = 0
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
                #print line_data
                
                if len(line_data) == len(Headder):
                    for idx,val in enumerate(line_data):
                        try:
                            Data[line_number-1,idx] = val
                        except:
#                             print "!"
                            pass
                    
                    if l1 != line_data[4]:
                        print line_data[1], line_data[4]
                        l1 = line_data[4]
                        
                        
                    
#     a= raw_input('prompt')
    print Headder
    return Data

if __name__ == '__main__':
    Data = get_rf_data()
    print Data
    xlim_lower = 100
    xlim_upper = 3000
    x = Data[:,1]
    y1 = Data[:,12]#there is data at 6 (~1280), 59 (~1320) and 112 (~1360)
    plt.subplot(1, 1, 1)
    plt.plot(x,y1)
    #plt.xlim(xlim_lower,xlim_upper)
    print y1
    plt.show()

#400 to 1200