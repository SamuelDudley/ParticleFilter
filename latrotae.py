import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f = 1.0 / 298.257223563		# WGS84
#f = (a-b)/a so... b = -(f*a - a)

a = WGS84_a = 6378137.0 			# metres
b = WGS84_b =  a- f*a
e =  eccentricity = math.sqrt(2*f-(f**2))

asq = a**2
esq = e**2


def WGS84EarthRadius(lat):
    # http://en.wikipedia.org/wiki/Earth_radius
    An = WGS84_a*WGS84_a * math.cos(lat)
    Bn = WGS84_b*WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )

def convertStoC(lat, lon, alt, h=0):
        x = math.cos(lat) * math.cos(lon) * (alt+h)
        y = math.cos(lat) * math.sin(lon) * (alt+h)
        z = math.sin(lat) * ((alt*(1-esq)) +h)# z is 'up'
        return (x,y,z)
x = []
y = []
z = []
(x1,y1,z1)=convertStoC(math.radians(90),math.radians(136.545941), WGS84EarthRadius(math.radians(90)))
(x2,y2,z2)=convertStoC(math.radians(-90),math.radians(136.545941), WGS84EarthRadius(math.radians(-90)))
(x3,y3,z3)=convertStoC(math.radians(-30),math.radians(136.545941), WGS84EarthRadius(math.radians(-30)))

print x1,y1,z1
print x1,y1,z2
print x1,y1,z3
x.append(x1)
x.append(x2)
x.append(x3)
y.append(y1)
y.append(y2)
y.append(y3)
z.append(z1)
z.append(z2)
z.append(z3)
fig = plt.figure(frameon=False)
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
ax.plot(x, y, z, color="r")
latgcsr = 80
latair = -20
plt.show()
latairm = 90+(latair)-(latgcsr)
if latairm >90:
    latairm= 90 - (latairm-90)
print latairm

latgcsm = 90+(latgcsr)-(latgcsr)
if latgcsm >90:
    latgcsm= 90 - (latgcsm-90)
print latgcsm


#plt.close("all")
