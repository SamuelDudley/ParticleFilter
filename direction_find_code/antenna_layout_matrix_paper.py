import numpy as np
import time

azimuth = 0.1# degrees
elevation = 80.2# degrees
print "azimuth:",np.radians(azimuth),(azimuth)," ","elevation:",np.radians(elevation), (elevation)

plotting = False
plotting_ans = True
linear_solve = True

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

def check_mask(a):
    if isinstance(a.mask, np.bool_):
        a.mask = np.ma.make_mask_none(a.shape)
        return a
    return a
        
    
    

# this is the 'center' of the antenna array
# create a large sphere around the centroid



# this is the antenna point
point1  = np.array([.0, 0.000, 0.000]) # one of three in a 'linear array'
point2  = np.array([0.1,0., 0.000]) # two of three in a 'linear array'
point3 = np.array([0.21, 0.0, 0.000]) # three of three in a 'linear array'
point4 = np.array([.15, .15,0.0000]) # extra antenna used to make unique soln's

c = 299792458 # speed of light [m/s]
frequency =1.3927e9 # 1.3 GHz

lamba = c/frequency # wavelength [m]

xx, yy = np.meshgrid(np.linspace(0.00001,2*np.pi,360), np.linspace(0.00001,np.pi,180), sparse=True)


# 2.0 >>>> 2 <<<<< 2.9
# z1_possible = np.modf((phase_calc(xx,yy,point2,point1,lamba))[1])
# z2_possible = np.modf((phase_calc(xx,yy,point3,point1,lamba))[1])
# z3_possible = np.modf((phase_calc(xx,yy,point4,point1,lamba))[1])
# z4_possible = np.modf((phase_calc(xx,yy,point2,point3,lamba))[1])
# z5_possible = np.modf((phase_calc(xx,yy,point2,point4,lamba))[1])
# z6_possible = np.modf((phase_calc(xx,yy,point3,point4,lamba))[1])

# 1.5 >>>> 2 <<<<< 2.4
z1_possible = np.rint(phase_calc(xx,yy,point2,point1,lamba))
z2_possible = np.rint(phase_calc(xx,yy,point3,point1,lamba))
z3_possible = np.rint(phase_calc(xx,yy,point4,point1,lamba))
z4_possible = np.rint(phase_calc(xx,yy,point2,point3,lamba))
z5_possible = np.rint(phase_calc(xx,yy,point2,point4,lamba))
z6_possible = np.rint(phase_calc(xx,yy,point3,point4,lamba))

z1_possible_range = np.unique(z1_possible)
z2_possible_range = np.unique(z2_possible)
z3_possible_range = np.unique(z3_possible)
z4_possible_range = np.unique(z4_possible)
z5_possible_range = np.unique(z5_possible)
z6_possible_range = np.unique(z6_possible)

z1_measured = phase_calc(np.radians(azimuth),np.radians(elevation),point2,point1,lamba)
z2_measured = phase_calc(np.radians(azimuth),np.radians(elevation),point3,point1,lamba)
z3_measured = phase_calc(np.radians(azimuth),np.radians(elevation),point4,point1,lamba)
z4_measured = phase_calc(np.radians(azimuth),np.radians(elevation),point2,point3,lamba)
z5_measured = phase_calc(np.radians(azimuth),np.radians(elevation),point2,point4,lamba)
z6_measured = phase_calc(np.radians(azimuth),np.radians(elevation),point3,point4,lamba)

print (np.modf(z1_measured)[0]), (np.modf(z1_measured)[1])
print (np.modf(z2_measured)[0]), (np.modf(z2_measured)[1])
print (np.modf(z3_measured)[0]), (np.modf(z3_measured)[1])
print (np.modf(z4_measured)[0]), (np.modf(z4_measured)[1])
print (np.modf(z5_measured)[0]), (np.modf(z5_measured)[1])
print (np.modf(z6_measured)[0]), (np.modf(z6_measured)[1])


A = np.array([(point2 - point1)[:-1],
              (point3 - point1)[:-1],
              (point4 - point1)[:-1],
              (point2 - point3)[:-1],
              (point2 - point4)[:-1],
              (point3 - point4)[:-1]]
             )

total = 0

import matplotlib.pyplot as plt

if linear_solve:        
    distance_list = []
    
    phase_component_12 = np.linspace(-1, 1, 50)
    
    z12_possible = np.modf((phase_calc(xx,yy,point2,point1,lamba)))[1]
    z13_possible = np.modf((phase_calc(xx,yy,point3,point1,lamba)))[1]
    z12_possible_range = np.unique(z12_possible)
    z13_possible_range = np.unique(z13_possible)
    z12_measured = phase_calc(np.radians(azimuth),np.radians(elevation),point2,point1,lamba)
    z13_measured = phase_calc(np.radians(azimuth),np.radians(elevation),point3,point1,lamba)
    measured_phase_12 = (np.modf(z12_measured)[0])
    measured_phase_13 = (np.modf(z13_measured)[0])
    max_geo_dist12 = geo_dist(point2, point1)
    max_geo_dist13 = geo_dist(point3, point1)
    
    phase_component_list = []
    for i12 in z12_possible_range:
        for i13 in z13_possible_range:
            phase_component_13 = ((max_geo_dist13/max_geo_dist12)*phase_component_12)+((max_geo_dist13/max_geo_dist12)*i12)-i13
            
            dist  = abs(measured_phase_12-(max_geo_dist12/max_geo_dist13)*measured_phase_13-(max_geo_dist12/max_geo_dist13)*i13+i12)/np.sqrt((-max_geo_dist12/max_geo_dist13)**2 + (1)**2)
            distance_list.append({'distance':dist, 'i12':i12, 'i13':i13})
            
            phase_component_list.append(phase_component_13)
            plt.plot(phase_component_13,phase_component_12, label = str(i12)+" "+str(i13))
            
            
        plt.scatter( measured_phase_13, measured_phase_12)
        
#         legend = plt.legend(loc='upper right')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
#     plt.show()
    
    from operator import itemgetter
    sorted_distance_list = sorted(distance_list, key=itemgetter('distance'), reverse=False)
    print sorted_distance_list
     
    if len(sorted_distance_list) > 1:
        certainty = 1 - (sorted_distance_list[0]['distance']/(sorted_distance_list[0]['distance']+sorted_distance_list[1]['distance']))
    else:
        certainty = 1
     
    print 'certainty', certainty*100, 'percent'



defatult_mask = np.ma.make_mask_none(z1_possible.shape)
space = 1
arr = None

for i in [sorted_distance_list[0]['i12']]:#z1_possible_range:#z1_possible_range:#:
    m1 = np.ma.masked_outside(z1_possible, i-space, i+space)
    check_mask(m1)
    z2_u = np.unique(z2_possible[~m1.mask].copy())
#     print 'z2', z2_u
         
    for ii in [sorted_distance_list[0]['i13']]:#z2_u:#z2_u:#z2_u:#[5]:#z2_possible_range:
        m2 = np.ma.masked_outside(z2_possible, ii-space, ii+space)
        check_mask(m2)
        m2_ = np.ma.mask_or(m1.mask, m2.mask)
        if m2_.shape != defatult_mask.shape:
            m2_ = defatult_mask
        z3_u = np.unique(z3_possible[~m2_].copy())
#         print 'z3', z3_u
    
#         plt.imshow(z3_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')    
#         print m2_ 
# #         plt.colorbar()              
#         plt.imshow(m2.mask, alpha=0.5, interpolation='none', cmap=plt.cm.Blues, aspect='auto') #this is the area left where the solution can lie...
#         plt.show()
 
        for iii in z3_u:#z3_u:#[4]:#z3_possible_range:
            m3 = np.ma.masked_outside(z3_possible, iii-space, iii+space)
            check_mask(m3)
            m3_ = np.ma.mask_or(m2_, m3.mask)
            z4_u = np.unique(z4_possible[~m3_].copy())
            
            for iiii in z4_u:#[1]:#z4_possible_range:
                m4 = np.ma.masked_outside(z4_possible, iiii-space, iiii+space)
                check_mask(m4)
                m4_ = np.ma.mask_or(m3_, m4.mask)
                z5_u = np.unique(z5_possible[~m4_].copy())
                
                for iiiii in z5_u:#z5_possible_range:
                    m5 = np.ma.masked_outside(z5_possible, iiiii-space, iiiii+space)
                    check_mask(m5)
                    m5_ = np.ma.mask_or(m4_, m5.mask)
                    z6_u = np.unique(z6_possible[~m5_].copy())
                    
                    if plotting:
                        plt.subplot(6, 1, 1)
                        plt.imshow(z1_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')
                        plt.subplot(6, 1, 2)
                        plt.imshow(z2_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')
                        plt.subplot(6, 1, 3)
                        plt.imshow(z3_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')
                        plt.subplot(6, 1, 4)
                        plt.imshow(z4_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')
                        plt.subplot(6, 1, 5)
                        plt.imshow(z5_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')
                        plt.subplot(6, 1, 6)
                        plt.imshow(z6_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')
                        
                        plt.imshow(~m5_, alpha=0.5, interpolation='none', cmap=plt.cm.Greys, aspect='auto') #this is the area left where the solution can lie... 
                         
                        plt.show()
 
                    phase_solution_space = np.array(np.meshgrid(
                                            i+np.modf(z1_measured)[0],
                                             ii+np.modf(z2_measured)[0],
                                             iii+np.modf(z3_measured)[0],
                                             iiii+np.modf(z4_measured)[0],
                                             iiiii+np.modf(z5_measured)[0],
                                             z6_u+np.modf(z6_measured)[0]
                                             )
                                 ).T.reshape(-1,6) # 6 is the number of indepentant solns 
                                 
                    if arr is None:
                        arr = phase_solution_space
                    else:
#                         print arr.shape, phase_solution_space.shape
                        arr = np.vstack((arr,phase_solution_space))
                        arr.shape
print 'solution space size:', arr.shape
                    
print 'arr',arr
t1 = time.time()
# raw_input('')

# phase_solution_space =  np.array(np.meshgrid(np.array(3)+np.modf(z1_measured)[0],
#                                              np.array(2)+np.modf(z2_measured)[0],
#                                              np.array(4)+np.modf(z3_measured)[0],
#                                              np.array(0)+np.modf(z4_measured)[0],
#                                              np.array(0)+np.modf(z5_measured)[0],
#                                              np.array(-1)+np.modf(z6_measured)[0]
#                                              )
#                                  ).T.reshape(-1,6) # 6 is the number of indepentant solns



# phase_solution_space =  np.array(np.meshgrid(z1_possible_range+np.modf(z1_measured)[0],
#                                              z2_possible_range+np.modf(z2_measured)[0],
#                                              z3_possible_range+np.modf(z3_measured)[0],
#                                              z4_possible_range+np.modf(z4_measured)[0],
#                                              z5_possible_range+np.modf(z5_measured)[0],
#                                              z6_possible_range+np.modf(z6_measured)[0]
#                                              )
#                                  ).T.reshape(-1,6) # 6 is the number of indepentant solns 
#                                  
# print phase_solution_space.shape

# b_mega = (lamba)*phase_solution_space
b_mega = (lamba)*arr
#these are only the measured values
# b = (lamba) *       np.array([np.modf(z1_measured)[0],
#                               np.modf(z2_measured)[0],
#                               np.modf(z3_measured)[0],
#                               np.modf(z4_measured)[0],
#                               np.modf(z5_measured)[0],
#                               np.modf(z6_measured)[0]])


# -ve values matter! if they are not correct then the angle is wrong! need to work out if the usrp will wrap back to pisitive or contiunue to report -ve 

b = (lamba) *       np.array([z1_measured,
                              z2_measured,
                              z3_measured,
                              z4_measured,
                              z5_measured,
                              z6_measured])
# print 'b_mega',b_mega
# print 'b',b



sol = np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,b))
sol_mega =  np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,b_mega.T))
# print 'sol_mega', sol_mega.T
print 'sol',sol
u_1 = sol[0]
v_1 = sol[1]

mat_mega = np.dot(A,sol_mega).T-b_mega

# print 'mat_mega', mat_mega

mat = A[:,0]*u_1 +  A[:,1]*v_1-b

# print 'mat',mat

sum = np.square(np.sum(mat, axis = 0))


sum_mega = np.square(np.sum(mat_mega, axis = 1))
# print 'sum_mega', sum_mega

print 'phase_sol', arr[np.argmin(sum_mega, axis=0)]
print 'min',sum_mega[sum_mega.argsort()[:20]]
itemindex = np.argmin(sum_mega, axis=0) #np.argmin(sum_mega, axis=0)#np.where(sum_mega==sum)
print 'single answer:', sum
print 'matrix answer:', sum_mega[itemindex], arr[itemindex], sol_mega.T[itemindex]
print arr[sum_mega.argsort()[:20]]




u_mega, v_mega = sol_mega.T[itemindex]
print u_mega, v_mega 
print u_1,v_1

az_mega = np.arctan2(v_mega,u_mega)
if az_mega < 0:
    az_mega += 2*np.pi
az =  np.arctan2(v_1,u_1)
if az < 0:
    az += 2*np.pi
print "-------------------"
print 'degrees:'
print 'az:', azimuth, 'sol:', np.degrees(az), np.degrees(az_mega)
print 'ele:', elevation, 'sol:', np.degrees(np.arcsin(u_1/np.cos(az))), np.degrees(np.arcsin(u_mega/np.cos(az_mega)))
print ""
print 'radians:'
print 'az:', np.radians(azimuth), 'sol:', az, az_mega
print 'ele:', np.radians(elevation), 'sol:', np.arcsin(u_1/np.cos(az)), np.arcsin(u_mega/np.cos(az_mega))
print "-------------------"

print 'time:', time.time()-t1
if plotting_ans:
    plot_amber = np.modf(arr[itemindex])[1]
    m1 = np.ma.masked_outside(z1_possible, plot_amber[0]-space, plot_amber[0]+space)
          
    m2 = np.ma.masked_outside(z2_possible, plot_amber[1]-space, plot_amber[1]+space)
    m2_ = np.ma.mask_or(m1.mask, m2.mask)

    m3 = np.ma.masked_outside(z3_possible, plot_amber[2]-space, plot_amber[2]+space)
    m3_ = np.ma.mask_or(m2_, m3.mask)

    m4 = np.ma.masked_outside(z4_possible, plot_amber[3]-space, plot_amber[3]+space)
    m4_ = np.ma.mask_or(m3_, m4.mask)
                     
    m5 = np.ma.masked_outside(z5_possible, plot_amber[4]-space, plot_amber[4]+space)
    m5_ = np.ma.mask_or(m4_, m5.mask)
    
    m6 = np.ma.masked_outside(z6_possible, plot_amber[5]-space, plot_amber[5]+space)
    m6_ = np.ma.mask_or(m5_, m6.mask)

#     plt.subplot(6, 1, 1)
#     plt.imshow(z1_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')
#     plt.subplot(6, 1, 2)
#     plt.imshow(z2_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')
#     plt.subplot(6, 1, 3)
#     plt.imshow(z3_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')
#     plt.subplot(6, 1, 4)
#     plt.imshow(z4_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')
#     plt.subplot(6, 1, 5)
#     plt.imshow(z5_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')
#     plt.subplot(6, 1, 6)
    plt.imshow(z5_possible, interpolation='none', cmap=plt.cm.Reds, aspect='auto')
                             
    plt.imshow(~m6_, alpha=0.5, interpolation='none', cmap=plt.cm.bwr, aspect='auto') #this is the area left where the solution can lie... 
    plt.scatter(np.degrees(az), np.degrees(np.arcsin(u_1/np.cos(az))), marker='+', s= 100, color="black")
    plt.scatter(np.degrees(az_mega), np.degrees(np.arcsin(u_mega/np.cos(az_mega))), marker='x', s= 200, color="blue")
    plt.xlim([0,360])
    plt.ylim([180,0])
                              
    plt.show()
         

         
         
