import numpy as np
import os
notam_dir = 'notam' #None
if notam_dir is not None:
    notam_path = os.path.join(os.getcwd(), notam_dir)
else:
    notam_path = os.getcwd()
    
for file in os.listdir(notam_path):
    if file.endswith(".plt"): # we assume this is a OziExplorer Track Point File
        print(file)
        area_name = file.split()[0].split('.')[0].lstrip().rstrip() # get prefix / area name from file
        print area_name
        data = np.loadtxt(os.path.join(notam_path,file),delimiter=',', skiprows=6, usecols=(0,1))
        print ' = ['
        for idx, row in enumerate(data):
            if idx + 1 != len(data):
                print "(",row[0]," , ", row[1], "),"
            else:
                print "(",row[0]," , ", row[1], ")"
        print "]"
        print ""

