import srtm
import Model
import numpy as np
import matplotlib.pyplot as plt

srtm_data = srtm.SrtmLayer()

sample_dist = 10.#m
lat = [-30.934841,-30.918156]
lon = [136.544633,136.540634]

f = 1.0 / 298.257223563
a = 6378137.0  
ele = []
(dist,forw,rev)= Model.vinc_dist(f,a,lat[0],lon[0],lat[1],lon[1])
dist = np.linspace(0, dist, 1000, True )
print dist
lat1 =lat[0]
lon1= lon[0]
for val in dist:
    (lat,lon,rev) = Model.vinc_pt(f,a,lat1,lon1,forw,val)
    ele.append(srtm_data.get_elevation(lat, lon))
#     print ele[-1]

plt.plot(dist,ele)
plt.show()

# 
# ele = srtm_data.get_elevation(lat, lon)
# print round(ele, 4)
 
 
 