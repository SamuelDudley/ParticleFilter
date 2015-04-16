import numpy as np
import math

num_points = 800
U = np.random.random(num_points)
V = np.random.random(num_points)

    
import math as m


def spherical_to_cartesian(vec):                                                  
    '''
    Convert spherical polar coordinates to cartesian coordinates:                       

    See the definition of spherical_cartesian_to_polar.                                 

    @param vec: A vector of the 3 polar coordinates (r, u, v)                           
    @return: (x, y, z)                                                                  
    ''' 
    (r, u, v) = vec                                                                     

    x = r * m.sin(u) * m.cos(v)  #can make an ellipse by adding a multiplier for r (in Y direction)                                                      
    y = r * m.sin(u) * m.sin(v)  #can make an ellipse by adding a multiplier for r (in Z direction)                                                       
    z = r * m.cos(u)           #can make an ellipse by adding a multiplier for r (in X direction)                                                         

    return [x, y, z]  

#radius = 1.
#points = np.array([spherical_to_cartesian([radius, 2 * np.pi * u, np.arccos(2*v - 1)]) for u,v in zip(U,V)])

radii = np.random.normal(0, 50, num_points)
points = np.array([spherical_to_cartesian([r, 2 * np.pi * u, np.arccos(2*v - 1)]) for r,u,v in zip(radii, U,V)])
#points = np.array([spherical_to_cartesian([r, 2 * np.pi * u,  0]) for r,u,v in zip(radii, U,V)])

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# values near to robbie's measurement => 1, further away => 0

fig, ax = plt.subplots()
x = np.linspace(-100, 100, 2000)
sigma2 = 10 ** 2
y= math.e ** -(x ** 2 / (2* sigma2))
ax.plot(x,y)
plt.show()

fig, ax = plt.subplots()
ax = Axes3D(fig)
ax.plot(points[:,0], points[:,1], points[:,2], 'x')
#ax.plot(points[:,0], points[:,2], 'x')
plt.show()