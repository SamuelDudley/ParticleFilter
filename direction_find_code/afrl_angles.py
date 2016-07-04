import numpy as np
azimuth = 90# degrees
elevation = 90 # degrees
azimuth = np.radians(azimuth)
elevation = np.radians(elevation)

d = 2
d1 = 3
# print np.sqrt(d1**2+(d)**2)
dist1 = ((d)*np.sin(azimuth)*np.cos(elevation))+(d1*np.sin(azimuth)*np.sin(elevation))
dist2 = ((d)*np.sin(azimuth)*np.cos(elevation))
dist3 = (d1*np.sin(azimuth)*np.sin(elevation))
print dist1, dist2, dist3