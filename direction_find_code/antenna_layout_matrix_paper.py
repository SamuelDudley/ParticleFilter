# we difine a planform in local x, y ,z Space
import numpy as np
import parasurf as ps

class Antenna(object):
    def __init__(self, id, location):
        self.location = location
        self.id = id
        self.x = location[0]
        self.y = location[1]
        self.z = location[2]
        
class Centroid(object):
    def __init__(self, id, points):
        self.id = id
        self.points = points
        self.number_of_points = len(points)
        self.ids = []
        for point in points:
            self.ids.append(point.id)
        self.location = None
        # do a check here..
        self.calculate()

    def calculate(self):
        x_sum = 0
        y_sum = 0
        z_sum = 0
        for point in self.points:
            x_sum += point.x
            y_sum += point.y
            z_sum += point.z
        
        self.x = x_sum / self.number_of_points
        self.y = y_sum / self.number_of_points
        self.z = z_sum / self.number_of_points
        self.location = (self.x, self.y, self.z)
        
# http://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion    
def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew
        
        
def pitch_point(location, angle):
    # a rotation around the x axis
    location.y = location.y*np.arccos(angle)
    location.z = point
    
    
def phase_calc(u,v,p1,p2,lamba):
    
    a = (((np.cos(u)*np.sin(v))*(p1[0] - p2[0])) + ((np.sin(u)*np.sin(v))*(p1[1] - p2[1])) + ((np.cos(v))*(p1[2]-p2[2])))
    b = np.sqrt((np.cos(u)*np.sin(v))**2.0 + (np.sin(u)*np.sin(v))**2.0 + (np.cos(v))**2.0)
    D1 = a/b
    return (D1/ lamba)
    
def geo_dist(p1,p2):
    _geo_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
#     _geo_dist = np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1]) + np.abs(p1[2] - p2[2])
    return _geo_dist
    
    
antennas = []


antennas.append(Antenna('1', (5,-5,0)))
antennas.append(Antenna('2', (10,-10,0)))
antennas.append(Antenna('1', (-5,-5,0)))
antennas.append(Antenna('2', (-10,-10,0)))

centroid = Centroid('c1', antennas)
(x,y,z) = centroid.location

# this is the 'center' of the antenna array
# create a large sphere around the centroid



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# this is the antenna point
point1  = np.array([.000, 0.000, 0.000]) # one of three in a 'linear array'
point2  = np.array([0.21,0.00, 0.000]) # two of three in a 'linear array'
point3 = np.array([0.0, 0.1500, 0.000]) # three of three in a 'linear array'
point4 = np.array([0.000, 0.22,0.00000]) # extra antenna used to make unique soln's

c = 299792458 # speed of light [m/s]
frequency = 1.350e9 # 1.3 GHz

lamba = c/frequency # wavelength [m]

azimuth = 50# degrees
elevation = 30# degrees
print "azimuth:",np.radians(azimuth)," ","elevation:",np.radians(elevation)

u, v = np.mgrid[0.00001:2*np.pi:100j, 0.00001:np.pi:100j]
u_measured, v_measured = np.mgrid[np.radians(azimuth):np.radians(azimuth):2j, np.radians(elevation):np.radians(elevation):2j]

z1_measured = phase_calc(u_measured,v_measured,point2,point1,lamba)
z2_measured = phase_calc(u_measured,v_measured,point3,point1,lamba)
z3_measured = phase_calc(u_measured,v_measured,point4,point1,lamba)
z4_measured = phase_calc(u_measured,v_measured,point2,point3,lamba)
z5_measured = phase_calc(u_measured,v_measured,point2,point4,lamba)
z6_measured = phase_calc(u_measured,v_measured,point3,point4,lamba)

print (np.modf(z1_measured[0][0])[0]), (np.modf(z1_measured[0][0])[1])
print (np.modf(z2_measured[0][0])[0]), (np.modf(z2_measured[0][0])[1])
print (np.modf(z3_measured[0][0])[0]), (np.modf(z3_measured[0][0])[1])
print (np.modf(z4_measured[0][0])[0]), (np.modf(z4_measured[0][0])[1])
print (np.modf(z5_measured[0][0])[0]), (np.modf(z5_measured[0][0])[1])
print (np.modf(z6_measured[0][0])[0]), (np.modf(z6_measured[0][0])[1])

A = np.array([[0.21, 0.00],
              [.0, 0.150],
              [0, 0.22],
              [0.21,-0.15],
              [0.21,-0.22],
              [0.0, 0.15-0.22]])

b = (lamba) *       np.array([np.modf(z1_measured[0][0])[0],
                              np.modf(z2_measured[0][0])[0],
                              np.modf(z3_measured[0][0])[0],
                              np.modf(z4_measured[0][0])[0],
                              np.modf(z5_measured[0][0])[0],
                              np.modf(z6_measured[0][0])[0]])


sol = np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,b))
print 'sol',sol
u_1 = sol[0]
v_1 = sol[1]

print u_1,v_1

az =  np.arctan2(v_1,u_1)
print az
print np.arcsin(u_1/np.cos(az))


z1 = phase_calc(u,v,point1,point2,lamba)
max_geo_dist1 = geo_dist(point1, point2)

z2 = phase_calc(u,v,point1,point3,lamba)
max_geo_dist2 = geo_dist(point1, point3)

z3 = phase_calc(u,v,point1,point4,lamba)
max_geo_dist3 = geo_dist(point1, point4)

z4 = phase_calc(u,v,point2,point3,lamba)
max_geo_dist4 = geo_dist(point2, point3)

z5 = phase_calc(u,v,point2,point4,lamba)
max_geo_dist5 = geo_dist(point2, point4)

z6 = phase_calc(u,v,point3,point4,lamba)
max_geo_dist6 = geo_dist(point3, point4)



print "z1", z1_measured[0][0]
print "z2", z2_measured[0][0]
print "z3", z3_measured[0][0]
print "z4", z4_measured[0][0]
print "z5", z5_measured[0][0]
print "z6", z6_measured[0][0]

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(u, v, z1,color="g") # full phase including intergers

plane_z = z1_measured[0][0]
plane_u, plane_v = np.mgrid[0.00001:2*np.pi:2j, 0.00001:np.pi:2j]
plane_z = np.mgrid[plane_z:plane_z:2j]
# ax.plot_surface(plane_u, plane_v, plane_z,color="k")

print "z1", plane_z




# ax = fig.gca(projection='3d')
# ax.plot_surface(u, v, z2,color="r") # full phase including intergers

plane_z = z2_measured[0][0]
plane_u, plane_v = np.mgrid[0.00001:2*np.pi:2j, 0.00001:np.pi:2j]
plane_z = np.mgrid[plane_z:plane_z:2j]
# ax.plot_surface(plane_u, plane_v, plane_z,color="k")

print "z2", plane_z
# plt.show()




# # get the points on the phase graph which correspond
# error_val = 0.01
# lower = sorted_distance_list[0]['i1']+measured_phase_1 - error_val
# upper = sorted_distance_list[0]['i1']+measured_phase_1 + error_val
# arr = np.ma.masked_outside(z1, lower, upper, copy=False)
# print np.ma.flatnotmasked_contiguous(arr)


#### parasurf ####
z1_slice_height = z1_measured[0][0]#-2+np.modf(z1_measured[0][0])[0]#z1_measured[0][0]#2+np.modf(z1_measured[0][0])[0]#z1_measured[0][0]
z2_slice_height = z2_measured[0][0]#-3+np.modf(z2_measured[0][0])[0]#z2_measured[0][0]#2+np.modf(z2_measured[0][0])[0]
z3_slice_height = z3_measured[0][0]#3+np.modf(z3_measured[0][0])[0]#z3_measured[0][0]
z4_slice_height = z4_measured[0][0]
z5_slice_height = z5_measured[0][0]
z6_slice_height = z6_measured[0][0]

xyz1 = np.array([u, v, z1])
srf1 = ps.ParaSurf(np.linspace(0, 1, len(u)),
                   np.linspace(0, 1, xyz1.shape[2]), xyz1)

# Create a test plane and cut the surface with it
ppt = np.array([1, 1, z1_slice_height]) # Point on plane
pn = np.array([0.0, 0.0, 1.0]) # Normal to plane
pn /= np.sqrt(np.dot(pn, pn)) # Create unit vector

D = np.dot(ppt, pn)
A, B, C = pn
planedef = np.array([A, B, C, D])

# Get the plane-surface intersection and plot
pipts1 = srf1.plane_intersect_pts(planedef, extend_srf=False)


xyz2 = np.array([u, v, z2])
srf2 = ps.ParaSurf(np.linspace(0, 1, len(u)),
                   np.linspace(0, 1, xyz2.shape[2]), xyz2)

# Create a plane and cut the surface with it
ppt = np.array([0, 0, z2_slice_height]) # Point on plane
pn = np.array([0.0, 0.0, 1.0]) # Normal to plane
pn /= np.sqrt(np.dot(pn, pn)) # Create unit vector

D = np.dot(ppt, pn)
A, B, C = pn
planedef = np.array([A, B, C, D])

# Get the plane-surface intersection and plot
pipts2 = srf2.plane_intersect_pts(planedef, extend_srf=False)


xyz3 = np.array([u, v, z3])
srf3 = ps.ParaSurf(np.linspace(0, 1, len(u)),
                   np.linspace(0, 1, xyz3.shape[2]), xyz3)

# Create a plane and cut the surface with it
ppt = np.array([0, 0, z3_slice_height]) # Point on plane
pn = np.array([0.0, 0.0, 1.0]) # Normal to plane
pn /= np.sqrt(np.dot(pn, pn)) # Create unit vector

D = np.dot(ppt, pn)
A, B, C = pn
planedef = np.array([A, B, C, D])

# Get the plane-surface intersection and plot
pipts3 = srf3.plane_intersect_pts(planedef, extend_srf=False)


xyz4 = np.array([u, v, z4])
srf4 = ps.ParaSurf(np.linspace(0, 1, len(u)),
                   np.linspace(0, 1, xyz4.shape[2]), xyz4)

# Create a plane and cut the surface with it
ppt = np.array([0, 0, z4_slice_height]) # Point on plane
pn = np.array([0.0, 0.0, 1.0]) # Normal to plane
pn /= np.sqrt(np.dot(pn, pn)) # Create unit vector

D = np.dot(ppt, pn)
A, B, C = pn
planedef = np.array([A, B, C, D])

# Get the plane-surface intersection and plot
pipts4 = srf4.plane_intersect_pts(planedef, extend_srf=False)


xyz5 = np.array([u, v, z5])
srf5 = ps.ParaSurf(np.linspace(0, 1, len(u)),
                   np.linspace(0, 1, xyz5.shape[2]), xyz5)

# Create a plane and cut the surface with it
ppt = np.array([0, 0, z5_slice_height]) # Point on plane
pn = np.array([0.0, 0.0, 1.0]) # Normal to plane
pn /= np.sqrt(np.dot(pn, pn)) # Create unit vector

D = np.dot(ppt, pn)
A, B, C = pn
planedef = np.array([A, B, C, D])

# Get the plane-surface intersection and plot
pipts5 = srf5.plane_intersect_pts(planedef, extend_srf=False)


xyz6 = np.array([u, v, z6])
srf6 = ps.ParaSurf(np.linspace(0, 1, len(u)),
                   np.linspace(0, 1, xyz6.shape[2]), xyz6)

# Create a plane and cut the surface with it
ppt = np.array([0, 0, z6_slice_height]) # Point on plane
pn = np.array([0.0, 0.0, 1.0]) # Normal to plane
pn /= np.sqrt(np.dot(pn, pn)) # Create unit vector

D = np.dot(ppt, pn)
A, B, C = pn
planedef = np.array([A, B, C, D])

# Get the plane-surface intersection and plot
pipts6 = srf6.plane_intersect_pts(planedef, extend_srf=False)


plt.scatter(pipts1[0].ravel(),pipts1[1].ravel(), color="black")
plt.scatter(pipts2[0].ravel(),pipts2[1].ravel(), color="red")
plt.scatter(pipts3[0].ravel(),pipts3[1].ravel(), color="green")
plt.scatter(pipts4[0].ravel(),pipts4[1].ravel(), color="blue")
plt.scatter(pipts5[0].ravel(),pipts5[1].ravel(), color="pink")
plt.scatter(pipts6[0].ravel(),pipts6[1].ravel(), color="orange")

plt.xlim([0,2*np.pi])
plt.ylim([0,np.pi])
plt.show()
#### parasurf ####


import numpy, time
from scipy import spatial as spat

def do_kdtree(combined_x_y_arrays,points):
    mytree = spat.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points, distance_upper_bound=0.1)
    print np.min(dist), points[np.argmin(dist)]
    return indexes

# Shoe-horn existing data for entry into KDTree routines
combined_x_y_arrays = numpy.dstack([pipts1[0].ravel(),pipts1[1].ravel()])[0]
points_list = numpy.dstack([pipts2[0].ravel(),pipts2[1].ravel()])[0]

results2 = do_kdtree(combined_x_y_arrays,points_list)

combined_x_y_arrays = numpy.dstack([pipts1[0].ravel(),pipts1[1].ravel()])[0]
points_list = numpy.dstack([pipts3[0].ravel(),pipts3[1].ravel()])[0]

results2 = do_kdtree(combined_x_y_arrays,points_list)


measured_phase_1 = (np.modf(z1_measured[0][0])[0]) # we can take the abs value here but it wont be the true value
measured_phase_2 = (np.modf(z2_measured[0][0])[0])
measured_phase_3 = (np.modf(z3_measured[0][0])[0])
measured_phase_4 = (np.modf(z4_measured[0][0])[0])
measured_phase_5 = (np.modf(z5_measured[0][0])[0])
measured_phase_6 = (np.modf(z6_measured[0][0])[0])

pairs_dict = {'z12':(z1,z2),
              'z13':(z1,z3),
              'z14':(z1,z4),
              'z15':(z1,z5),
              'z16':(z1,z6),
              'z23':(z2,z3),
              'z24':(z2,z4),
              'z25':(z2,z5),
              'z26':(z2,z6),
              'z34':(z3,z4),
              'z35':(z3,z5),
              'z36':(z3,z6),
              'z45':(z4,z5),
              'z46':(z4,z6),
              'z56':(z5,z6)
              }
results_dict = {}
for key in pairs_dict.keys():
    (zx,zy) = pairs_dict[key]
    i1_max = int(np.max(np.modf(zx)[1]))
    i1_min = int(np.min(np.modf(zx)[1]))
    # print 'sin:', (max_geo_dist1/lamba)*np.sin(np.max(u)) # this does not work for all cases...
    i2_max =int(np.max(np.modf(zy)[1])) # this does...
    i2_min =int(np.min(np.modf(zy)[1])) # this does...
#     print i1_max, i2_max, i1_min, i2_min
    
    
    z1_int_max = int(np.max(np.modf(zx)[1]))
    z1_int_min = int(np.min(np.modf(zx)[1]))
#     print z1_int_min, z1_int_max
    
    i1_dict={}
    for i in range(z1_int_min,z1_int_max+1):
        m = np.ma.masked_where(zx.astype(int) != i, zy.astype(int))
        i1_dict[str(i)] = np.unique(zy.astype(int)[~m.mask].copy())
    results_dict[key] = i1_dict

# for key in results_dict.keys():
#     print key
#     for ent in results_dict[key].keys():
#         print ent, results_dict[key][ent]
#     print ""
results_dict_possible = {}

z1_possible = np.array([-1])
z2_possible  = np.array([-1])
z3_possible  = np.array([])
z4_possible  = np.array([])
z5_possible  = np.array([])
z6_possible  = np.array([])

# key = "z12"
# print 'key', np.asarray(results_dict[key].keys(), dtype=int)
# for ambiguity in results_dict[key].keys():
#  
#     if ambiguity in z1_possible.astype(str):
#        z2_possible = np.append(z2_possible,results_dict[key][ambiguity]).astype(int)
#        z2_possible = np.unique(z2_possible)
        
key = "z23"
for ambiguity in results_dict[key].keys():
    if ambiguity in z2_possible.astype(str):
       z3_possible= np.append(z3_possible,results_dict[key][ambiguity]).astype(int)
       z3_possible = np.unique(z3_possible)
       
key = "z34"
for ambiguity in results_dict[key].keys():
    if ambiguity in z3_possible.astype(str):
       z4_possible= np.append(z4_possible,results_dict[key][ambiguity]).astype(int)
       z4_possible = np.unique(z4_possible)

key = "z45"
for ambiguity in results_dict[key].keys():
    if ambiguity in z4_possible.astype(str):
       z5_possible= np.append(z5_possible,results_dict[key][ambiguity]).astype(int)
       z5_possible = np.unique(z5_possible)

key = "z56"
for ambiguity in results_dict[key].keys():
    if ambiguity in z5_possible.astype(str):
       z6_possible= np.append(z6_possible,results_dict[key][ambiguity]).astype(int)
       z6_possible = np.unique(z6_possible)
       
print "z1", z1_possible
print "z2", z2_possible
print "z3", z3_possible
print "z4", z4_possible
print "z5", z5_possible
print "z6", z6_possible

z1_int_max = int(np.max(np.modf(z1)[1]))
z1_int_min = int(np.min(np.modf(z1)[1]))
print z1_int_max, z1_int_min
# phase_component_2 = np.linspace(-1, 1, 100)
# plt.xlim([-1,1])
# plt.ylim([-1,1])
# phase_component_list = []
# for i1 in range(z1_int_min,z1_int_max+1):#range(2,3):#range(z1_int_min,z1_int_max+1):
#     for i2 in i1_dict[str(i1)]:#range(-1,0):#i1_dict[str(i1)]:
#         phase_component_1 = ((z1/z2)*phase_component_2)+((z1/z2)*i2)-i1
#         
#         plt.plot(phase_component_2,phase_component_1, label = str(i1)+" "+str(i2))
#         
#         
#     plt.scatter( measured_phase_2, measured_phase_1)
#     
# 
# legend = plt.legend(loc='upper right')
# print  measured_phase_2, measured_phase_1
# 
# plt.show()

distance_list = []


phase_component_2 = np.linspace(-1, 1, 50)
plt.xlim([-1,1])
plt.ylim([-1,1])
phase_component_list = []
for i1 in np.asarray(results_dict['z12'].keys(), dtype=int):
    for i2 in np.asarray(results_dict['z23'].keys(), dtype=int):
        phase_component_1 = ((max_geo_dist1/max_geo_dist2)*phase_component_2)+((max_geo_dist1/max_geo_dist2)*i2)-i1
        
        dist  = abs(measured_phase_1-(max_geo_dist1/max_geo_dist2)*measured_phase_2-(max_geo_dist1/max_geo_dist2)*i2+i1)/np.sqrt((-max_geo_dist1/max_geo_dist2)**2 + (1)**2)
        distance_list.append({'distance':dist, 'i1':i1, 'i2':i2})
        #print dist, i1, i2 
        
        phase_component_list.append(phase_component_2)
        plt.plot(phase_component_2,phase_component_1, label = str(i1)+" "+str(i2))
        
        
    plt.scatter( measured_phase_2, measured_phase_1)
    
    legend = plt.legend(loc='upper right')
plt.show()

from operator import itemgetter
sorted_distance_list = sorted(distance_list, key=itemgetter('distance'), reverse=False)
print sorted_distance_list

if len(sorted_distance_list) > 1:
    certainty = 1 - (sorted_distance_list[0]['distance']/(sorted_distance_list[0]['distance']+sorted_distance_list[1]['distance']))
else:
    certainty = 1

print 'certainty', certainty*100, 'percent'


