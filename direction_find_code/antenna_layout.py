# we difine a planform in local x, y ,z Space
import numpy as np

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
theta = np.pi/3.0
gamma = np.pi/2.0
point  = np.array([0, 0, 0])
normal = np.array([np.sin(theta)*np.cos(gamma),np.sin(theta)*np.sin(gamma), np.cos(gamma)])



# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)

# create x,y
xx, yy = np.meshgrid(range(10), range(10))

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]


# this is the antenna point
point2  = np.array([0.10, 0.0, 0.0])
point3 = np.array([0.21,0.0,0.0])
point4  = np.array([-0.1, 0.1, 0.1])
point5 = np.array([0.2,0.2,.1])

c = 299792458 # speed of light [m/s]
frequency = 6*10**9.0 # 1.3 GHz

lamba = c/frequency # wavelength [m]


# # calculate the distance from the plane to the point
# D = ((np.sin(theta)*np.cos(gamma))*point2[0] + (np.sin(theta)*np.sin(gamma))*point2[1] + (np.cos(gamma))*point2[2])/np.sqrt((np.sin(theta)*np.cos(gamma))**2.0 + (np.sin(theta)*np.sin(gamma))**2.0 + (np.cos(gamma))**2.0)
# print D
# print np.modf(D/lamba)
# dist = np.modf(D/lamba)[0] # get the fractional component of the answer
# 
# 
# 
# # plot the surface
# plt3d = plt.figure().gca(projection='3d')
# plt3d.plot_surface(xx, yy, z)
# # plot the antenna point
# plt3d.scatter(point2[0],point2[1],point2[2])

# plt.show()


# create the soln space
# x range [0, 2 pi]
# y range [0, pi]

# theta = np.linspace(0, 2*np.pi, 400)
# gamma = np.linspace(0, np.pi/2, 400)

#draw sphere
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_aspect("equal")
angle = 0.9
u, v = np.mgrid[0.00001:2*np.pi:800j, 0.00001:np.pi:400j]#0.00001:np.pi/2:100j]
# x=np.cos(u)*np.sin(v)
# y=np.sin(u)*np.sin(v)
# z=np.cos(v)
# print x.shape, y.shape, z.shape
# ax.plot_wireframe(x, y, z, color="r")
# xx, yy = np.meshgrid(theta,gamma)
p1 = point
p2 = point2

a = (((np.cos(u)*np.sin(v))*(p1[0] - p2[0])) + ((np.sin(u)*np.sin(v))*(p1[1] - p2[1])) + ((np.cos(v))*(p1[2]-p2[2])))
b = np.sqrt((np.cos(u)*np.sin(v))**2.0 + (np.sin(u)*np.sin(v))**2.0 + (np.cos(v))**2.0)
D1 = a/b

max_geo_dist1 = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
print max_geo_dist1

p1 = point
p2 = point3

c = (((np.cos(u)*np.sin(v))*(p1[0] - p2[0])) + ((np.sin(u)*np.sin(v))*(p1[1] - p2[1])) + ((np.cos(v))*(p1[2]-p2[2])))
d = np.sqrt((np.cos(u)*np.sin(v))**2.0 + (np.sin(u)*np.sin(v))**2.0 + (np.cos(v))**2.0)
D2 = c/d

max_geo_dist2 = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
print max_geo_dist2

# D1/D2 is the same as a/c


# d_norm_phase1 = (a/c)*D2/lamba)/2*np.pi)[0] + (a/c)*np.modf((D2/lamba)/2*np.pi)[1] - np.modf((D1/lamba)/2*np.pi)[1]
# https://www.wolframalpha.com/input/?i=sin%282*pi*%28%28%28cos%28u%29*sin%28v%29*a%29+%2B+%28sin%28u%29*sin%28v%29*b%29+%2B+%28cos%28v%29*c%29%29%2F%28sqrt%28%28cos%28u%29*sin%28v%29%29**2.0+%2B+%28sin%28u%29*sin%28v%29%29**2.0+%2B+%28cos%28v%29%29**2.0%29%29%2Fl%29%29
# sin((2 pi (a cos(u) sin(v)+b sin(u) sin(v)+c cos(v)))/(l sqrt((sin(u) sin(v))^2.+(cos(u) sin(v))^2.+cos^2.(v))))
# z= np.modf(D/lamba)[0]
z= (D1/lamba)#-np.modf(D/lamba)[1]

fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.set_xlim3d(0, 10)
# ax.set_ylim3d(0, 10)
# ax.set_zlim3d(0, 10)
# ax.set_aspect("equal")
r1 = 2+np.sin((2*np.pi*D1)/lamba)
# # r1 = D1
x1=r1*np.cos(u)*np.sin(v)
y1=r1*np.sin(u)*np.sin(v)
z1=r1*np.cos(v)
# ax.plot_wireframe(x1, y1, z1)
r2 = 2+np.sin((2*np.pi*D2)/lamba)
# # r2 = D2
x2=r2*np.cos(u)*np.sin(v)
y2=r2*np.sin(u)*np.sin(v)
z2=r2*np.cos(v)
# ax.plot_wireframe(x2, y2, z2,color="r")


# r3 = 2+np.sin((D1/D2)*(((2*np.pi*D2)/lamba)+1))
r3 = 2+np.sin((D1/D2)*((2*np.pi*D2)/lamba))

# # r2 = D2
x3=r3*np.cos(u)*np.sin(v)
y3= r3*np.sin(u)*np.sin(v)
z3=r3*np.cos(v)
# ax.plot_wireframe(x3, y3, z3,color="k")
# # r2 = D2
# x2=r2*np.cos(u)*np.sin(v)
# y2=r2*np.sin(u)*np.sin(v)
# z2=r2*np.cos(v)
# ax.plot_wireframe(x2, y2, z2,color="r")

# ax.plot_surface(u, v, 2*np.pi*((D2/lamba) - np.modf(D2/lamba)[1]) ,color="b")
# ax.plot_surface(u, v, 2*np.pi*((D1/lamba) - np.modf(D1/lamba)[1]) ,color="r")
# ax.plot_wireframe(u, v, np.sin((2*np.pi*D1)/lamba) ,color="r") 

# ax.plot_surface(u, v, np.sin((2*np.pi*D2)/lamba) ,color="b")          

ax.plot_wireframe(u, v, D2/lamba,color="g") # full phase including intergers

# ax.plot_surface(u, v, (D1/D2) ,color="r")
# ax.plot_surface(u, v,np.modf((2*np.pi/lamba)*D1)[0])
plt.show()


#http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2896220/

# 
# # Hopf coordinates for SO(3) (4d sphere)
# # u, v, y = np.mgrid[0.00001:2*np.pi:200j, 0.00001:np.pi/2:100j, 0.00001:2*np.pi:200j]
# # here u is the azmeth, v the elevation and y the phase
# # x1 = np.cos(v/2)*np.cos(np.pi*(D1/lamba)) # here y/2 == (2*np.pi*(D1/lamba)))/2 == np.pi*(D1/lamba)))
# # x2 = np.cos(v/2)*np.sin(np.pi*(D1/lamba))
# # x3 = np.sin(v/2)*np.cos(u+(np.pi*(D1/lamba)))
# # x4 = np.sin(v/2)*np.sin(u+(np.pi*(D1/lamba)))
# # 
# # print x1
# # print x2
# # print x3
# # print x4
#
i1_max = int(np.max(np.modf(D1/lamba)[1]))
print 'sin:', (max_geo_dist1/lamba)*np.sin(np.max(u)) # this does not work for all cases...
i2_max =int(np.max(np.modf(D2/lamba)[1])) # this does...
print i1_max, i2_max
x = np.linspace(0, 1, 50)
plt.xlim([0,1])
plt.ylim([0,1])
for i1 in range(i1_max):
    for i2 in range(i2_max):
        y = (max_geo_dist1/max_geo_dist2)*x+(max_geo_dist1/max_geo_dist2)*i2-i1
        plt.plot(x,y)
plt.show()


