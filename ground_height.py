#http://gis.stackexchange.com/questions/30448/local-coordinate-to-geocentric
"""need to get the location of the gcs and aaircraft in cart coords. as light travels in streight lines!"""


import srtm
import Model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import sin, cos

from numpy import array, dot, arccos, clip
from numpy.linalg import norm

srtm_data = srtm.SrtmLayer()




def latlonalt_to_xyz(lat, lon, alt, north = 0, east = 0):    
    f = np.radians(lon)
    q = np.radians(lat)
    rot = np.array(
                   [[-sin(f),   -cos(f)*sin(q),     cos(f)*cos(q)],
                    [cos(f),    -sin(f)*sin(q),     sin(f)*cos(q)],
                    [0,         cos(q),             sin(q)       ]]
                   )
    """
    first col is a unit-length vector giving the local x (east) direction in terms of the geocentric coordinates.
    The second column is a unit vector giving the local y (north) direction.
    The third column is the local outward unit normal, or "up" direction.
    """
    
    b = 6356752.3142
    a = 6378137
    w = np.arctan(b/a*np.tan(q))
    
    rot2 = np.array(
                    [[-sin(f),   -cos(f)*sin(q),     cos(f)*cos(q),      a*cos(w)*cos(f)],
                     [cos(f),    -sin(f)*sin(q),     sin(f)*cos(q),      a*cos(w)*sin(f)],
                     [0,         cos(q),             sin(q),              b*sin(w)  ],
                     [0,         0,                  0,                 1]])


    loc1 = np.array([north,east,alt,1])
    
    #calc the vector from this point to the center of the earth model...
    
    return rot2.dot(loc1.T)[:3]



fig = plt.figure()
ax = fig.add_subplot(111)#, projection='3d')

f = 1.0 / 298.257223563
a = 6378137.0 

lat = -30.934841
lon = 136.544633

srtm_alt = srtm_data.get_elevation(lat, lon)
xyz = latlonalt_to_xyz(lat=lat, lon = lon, alt = srtm_alt+10)
u = xyz
(lat,lon,rev) = Model.vinc_pt(f,a,lat,lon,0,50000)
srtm_alt = srtm_data.get_elevation(lat, lon)
xyz = latlonalt_to_xyz(lat=lat, lon = lon, alt = srtm_alt+500)
v = xyz
c = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle
angle = arccos(clip(c, -1, 1)) # if you really want the angle
print np.degrees(angle)

print Model.cartDistance(u,v) #this is the RF distance (slightly greater than great circle)

'''


sample_dist = 10.#m
lat = [-30.934841,-30.918156]
lon = [136.544633,136.540634]
# lat = [36.234600,36.4345]
# lon = [-108.619987,-108.419989]

ele = []
(dist,forw,rev)= Model.vinc_dist(f,a,lat[0],lon[0],lat[1],lon[1])
print dist
dist = np.linspace(0, dist, 2, True )
lat1 =lat[0]
lon1= lon[0]
for val in dist:
    (lat,lon,rev) = Model.vinc_pt(f,a,lat1,lon1,forw,val)
    print lat,lon
    srtm_alt = srtm_data.get_elevation(lat, lon)
    xyz = latlonalt_to_xyz(lat=lat, lon = lon, alt = srtm_alt)
    x,y,z = xyz
    #ax.scatter(x,y,z) #3d
    ax.scatter(val,srtm_alt) #2d
    print xyz



plt.show()
# plt.show()
# 
# # 
# # ele = srtm_data.get_elevation(lat, lon)
# # print round(ele, 4)
'''
 