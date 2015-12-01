import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




WGS84_f = 1.0 / 298.257223563		# WGS84
round_f = 0

#f = (a-b)/a so... b = -(f*a - a)

WGS84_a = 6378137.0 			# metres
round_a = 6378137.0
WGS84_b =  WGS84_a- WGS84_f*WGS84_a
round_b = round_a- round_f*round_a
WGS84_e = math.sqrt(2*WGS84_f-(WGS84_f**2)) #eccentricity
round_e = math.sqrt(2*round_f-(round_f**2))

WGS84_asq = WGS84_a**2
WGS84_esq = WGS84_e**2

round_asq = round_a**2
round_esq = round_e**2



antennaGCSLat = math.radians(-34.407625)#-30.934901)
antennaGCSLon = math.radians(138.556481)#136.544666)

#need to take alt diff into account when at larger dist...
#eg. actual alt is preceved as lower, because of the curve of the earth...



def sin2(x):
    return math.sin(x)**2.0

def cos2(x):
    return math.cos(x)**2.0

def freeSpaceLoss(lamba,Pt,Gt,Gr,f,d):
    #Pr = Pt*Gt*Gr*(lamba/(4*math.pi*d))**2
    Pr = -147.6+20*np.log10(d)+20*np.log10(f)
    return -Pr#10*np.log10(1000*Pr/1)
"""
def twoRayModel(Gt, Gr, ht, hr, d):
    #Lp = -10*np.log10(Gt)-10*np.log10(Gr)-20*np.log10(ht)-20*np.log10(hr)+40*np.log10(d)
    Lp = 40*np.log10(d)-20*np.log10(ht)-20*np.log10(hr)
    return Lp
"""

def twoRayModel1(lamba,Pt, Gt, Gr, ht, hr, d):
    Pr = Pt*Gt*Gr*((lamba**2.0)/(4*math.pi*d)**2)*4*cos2(((2*math.pi)/lamba)*((ht*hr)/d))
    R = 1 #reflection loss
    beta = (2*math.pi)/lamba
    #Pr =((lamba/(4*math.pi*d))**2)*abs(1+R*cmath.exp(1j*beta*((2*ht*hr)/d)))**2
    #Pr = Pt*Gt*Gr*2*((lamba/(4*math.pi*d))**2)*(1-math.cos((2*beta*ht*hr)/d))
    #Pr = -147.6+20*np.log10(d)
    return 10*np.log10((Pr))

def twoRayModel2(lamba,Pt, Gt, Gr, ht, hr, d):
    Pr = Pt*Gt*Gr*((lamba**2.0)/(4*math.pi*d)**2)*4*sin2(((2*math.pi)/lamba)*((ht*hr)/d))
    R = 1 #reflection loss
    beta = (2*math.pi)/lamba
    #Pr =((lamba/(4*math.pi*d))**2)*abs(1+R*cmath.exp(1j*beta*((2*ht*hr)/d)))**2
    #Pr = Pt*Gt*Gr*2*((lamba/(4*math.pi*d))**2)*(1-math.cos((2*beta*ht*hr)/d))
    #Pr = -147.6+20*np.log10(d)
    return 10*np.log10((Pr))


def twoRayModel3(Pt, Gt, Gr, ht, hr, d):
    #Pr = Pt*Gt*Gr*((ht*hr)/(d**2.0))**2.0
    #Pr = ((Pt*Gt*Gr*(ht**2.0)*(hr**2.0))/(d**4.0))
    #Pr = (1/(Gr*Gt))*((d**4.0)/((ht*hr)**2))
    Pr = 40*np.log10(d)-20*np.log10(ht)-20*np.log10(hr)
    return -Pr#10*np.log10(1000*Pr/1)

def criticalDist(lamba, ht, hr):
    cd = (4*hr*ht)/lamba
    return cd
    
def getAircraftData():
    inputFile = "Short_pulled.hawk"
    fid = open(inputFile, 'r')
    headder = fid.readline()

    data = []
    for line in fid:
        data.append(([x.rstrip('\n') for x in line.split(' ') if x.lstrip('#[').rstrip(']\n')!= '']))
    return data

def vinc_dist(  f,  a,  phi1,  lembda1,  phi2,  lembda2 ) :
        """ 
        Returns the distance between two geographic points on the ellipsoid
        and the forward and reverse azimuths between these points.
        lats, longs and azimuths are in decimal degrees, distance in metres 

        Returns ( s, alpha12,  alpha21 ) as a tuple
        """
        
        phi1 = math.degrees(phi1)
        phi2 = math.degrees(phi2)
        lembda1 = math.degrees(lembda1)
        lembda2 = math.degrees(lembda2)
        
        
        if (abs( phi2 - phi1 ) < 1e-8) and ( abs( lembda2 - lembda1) < 1e-8 ) :
                return 0.0, 0.0, 0.0

        piD4   = math.atan( 1.0 )
        two_pi = piD4 * 8.0

        phi1    = phi1 * piD4 / 45.0
        lembda1 = lembda1 * piD4 / 45.0		# unfortunately lambda is a key word!
        phi2    = phi2 * piD4 / 45.0
        lembda2 = lembda2 * piD4 / 45.0

        b = a * (1.0 - f)

        TanU1 = (1-f) * math.tan( phi1 )
        TanU2 = (1-f) * math.tan( phi2 )

        U1 = math.atan(TanU1)
        U2 = math.atan(TanU2)

        lembda = lembda2 - lembda1
        last_lembda = -4000000.0		# an impossibe value
        omega = lembda

        # Iterate the following equations, 
        #  until there is no significant change in lembda 

        while ( last_lembda < -3000000.0 or lembda != 0 and abs( (last_lembda - lembda)/lembda) > 1.0e-9 ) :

                sqr_sin_sigma = pow( math.cos(U2) * math.sin(lembda), 2) + \
                        pow( (math.cos(U1) * math.sin(U2) - \
                        math.sin(U1) *  math.cos(U2) * math.cos(lembda) ), 2 )

                Sin_sigma = math.sqrt( sqr_sin_sigma )

                Cos_sigma = math.sin(U1) * math.sin(U2) + math.cos(U1) * math.cos(U2) * math.cos(lembda)
        
                sigma = math.atan2( Sin_sigma, Cos_sigma )

                Sin_alpha = math.cos(U1) * math.cos(U2) * math.sin(lembda) / math.sin(sigma)
                alpha = math.asin( Sin_alpha )

                Cos2sigma_m = math.cos(sigma) - (2 * math.sin(U1) * math.sin(U2) / pow(math.cos(alpha), 2) )

                C = (f/16) * pow(math.cos(alpha), 2) * (4 + f * (4 - 3 * pow(math.cos(alpha), 2)))

                last_lembda = lembda

                lembda = omega + (1-C) * f * math.sin(alpha) * (sigma + C * math.sin(sigma) * \
                        (Cos2sigma_m + C * math.cos(sigma) * (-1 + 2 * pow(Cos2sigma_m, 2) )))

        u2 = pow(math.cos(alpha),2) * (a*a-b*b) / (b*b)

        A = 1 + (u2/16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))

        B = (u2/1024) * (256 + u2 * (-128+ u2 * (74 - 47 * u2)))

        delta_sigma = B * Sin_sigma * (Cos2sigma_m + (B/4) * \
                (Cos_sigma * (-1 + 2 * pow(Cos2sigma_m, 2) ) - \
                (B/6) * Cos2sigma_m * (-3 + 4 * sqr_sin_sigma) * \
                (-3 + 4 * pow(Cos2sigma_m,2 ) )))

        s = b * A * (sigma - delta_sigma)

        alpha12 = math.atan2( (math.cos(U2) * math.sin(lembda)), \
                (math.cos(U1) * math.sin(U2) - math.sin(U1) * math.cos(U2) * math.cos(lembda)))

        alpha21 = math.atan2( (math.cos(U1) * math.sin(lembda)), \
                (-math.sin(U1) * math.cos(U2) + math.cos(U1) * math.sin(U2) * math.cos(lembda)))

        if ( alpha12 < 0.0 ) : 
                alpha12 =  alpha12 + two_pi
        if ( alpha12 > two_pi ) : 
                alpha12 = alpha12 - two_pi

        alpha21 = alpha21 + two_pi / 2.0
        if ( alpha21 < 0.0 ) : 
                alpha21 = alpha21 + two_pi
        if ( alpha21 > two_pi ) : 
                alpha21 = alpha21 - two_pi

        alpha12    = alpha12    * 45.0 / piD4
        alpha21    = alpha21    * 45.0 / piD4
        return s#, alpha12,  alpha21 

   # END of Vincenty's Inverse formulae

   
# Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
def earthRadius(lat,a,b):
    # http://en.wikipedia.org/wiki/Earth_radius
    An = a*a * math.cos(lat)
    Bn = b*b * math.sin(lat)
    Ad = a * math.cos(lat)
    Bd = b * math.sin(lat)
    return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )



def cartDistance(x1,y1,z1,x2,y2,z2):
        dist = math.sqrt(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))
        return dist

def convertStoC(lat, lon, esq, alt, h):
        x = math.cos(lat) * math.cos(lon) * (alt+h)
        y = math.cos(lat) * math.sin(lon) * (alt+h)
        z = math.sin(lat) * ((alt*(1-esq)) +h)# z is 'up'
        return (x,y,z)

def applyLatOffset(latr,latf):
    latr = math.degrees(latr)
    latf = math.degrees(latf)
    latmod = 90+(latr)-(latf)
    if latmod >90:
        latmod= 90 - (latmod-90)
    return math.radians(latmod)
    


"""
data = getAircraftData()
#EarthRadius = 6371009
latArray = [float(x[1]) for x in data]
lonArray = [float(x[2]) for x in data]
altArray = [float(x[3])-129.077 for x in data]
rssi2Array = [float(x[7]) for x in data]
#rssi3Array = [float(x[8]) for x in data]
del data
"""
latArray=np.linspace(-34.4,-34.4,100)#-30.929952,-30.463819, 100)
lonArray = np.linspace(138.5,138.6,100)#136.545213, 136.422475, 100)
altArray = np.linspace(120,120,100)

distArrayS = [ vinc_dist(WGS84_f,WGS84_a,antennaGCSLat,antennaGCSLon,math.radians(latArray[x]),math.radians(lonArray[x])) for x in range(len(latArray))]

altArrayElipMod=[]
altArrayElipDif=[]

altArrayRounMod=[]
altArrayRounDif=[]

alt1Array = []
xe = []
ye = []
ze = []
xr = []
yr = []
zr = []
zdiffe = []
zdiffr = []

distToAdde=[]
distToAddr=[]
distArraySe = []
distArraySr = []
h2 = 3#10.55
for i in range(len(altArray)):
    alt2= earthRadius(antennaGCSLat, WGS84_a, WGS84_b)#EarthRadius#
    
    alt1=earthRadius(math.radians(latArray[i]), WGS84_a, WGS84_b)#EarthRadius#
    h1 = altArray[i]
    alt1Array.append(alt1)

    (x1, y1, z1) = convertStoC(applyLatOffset(math.radians(latArray[i]),antennaGCSLat),math.radians(lonArray[i]),WGS84_esq,alt1,h1)
    (x2, y2, z2) = convertStoC(applyLatOffset(antennaGCSLat,antennaGCSLat),antennaGCSLon,WGS84_esq,alt2,h2)

    xe.append(x1)
    ye.append(y1)
    ze.append(z1)
    #make gcs to aircraft vector
    v1x = (x2-x1)
    v1y =  (y2-y1)
    v1z = (z2-z1)
    zdiffe.append(z1-z2)
    #make gcs to center of earth vector
    v2x = (0-x2)
    v2y = (0-y2)
    v2z = (0-z2)

    
    extraDist = math.fabs(math.sin(antennaGCSLat-math.radians(latArray[i]))*altArray[i])
    distToAdde.append(extraDist)
    distArraySe.append(distArrayS[i]+extraDist)

    
    #dot product
    dot = (v1x*v2x)+(v1y*v2y)+(v1z*v2z)

    sqrt1 = math.sqrt((v1x**2)+(v1y**2)+(v1z**2))
    sqrt2 = math.sqrt((v2x**2)+(v2y**2)+(v2z**2))
    try:
        angle = math.degrees(math.acos(dot/(sqrt1*sqrt2)))
    except:
        print dot,sqrt1,sqrt2,dot/(sqrt1*sqrt2)
    ele = 90-angle
    


    #sin(theta) = opp / hyp = altdiff / cartdist
    #so the elevation change with dist is... 
                        
    altDiff = math.sin(math.radians(ele))*cartDistance(x1,y1,z1,x2,y2,z2)
    #calc'd alt is alt2+h2+altDiff
    altArrayElipMod.append(h2+altDiff)
    #print altDiff
############################################################################
    alt2= earthRadius(antennaGCSLat, round_a, round_b)#EarthRadius#
    
    alt1=earthRadius(math.radians(latArray[i]), round_a, round_b)#EarthRadius#
    h1 = altArray[i]
    alt1Array.append(alt1)

    (x1, y1, z1) = convertStoC(applyLatOffset(math.radians(latArray[i]),antennaGCSLat),math.radians(lonArray[i]),round_esq,alt1,h1)
    (x2, y2, z2) = convertStoC(applyLatOffset(antennaGCSLat,antennaGCSLat),antennaGCSLon,round_esq,alt2,h2)
    zdiffr.append(z1-z2+h2)
    xr.append(x1)
    yr.append(y1)
    zr.append(z1)
    #make gcs to aircraft vector
    v1x = (x2-x1)
    v1y =  (y2-y1)
    v1z = (z2-z1)

    #make gcs to center of earth vector
    v2x = (0-x2)
    v2y = (0-y2)
    v2z = (0-z2)

    extraDist = math.fabs(math.sin(antennaGCSLat-math.radians(latArray[i]))*altArray[i])
    distToAddr.append(extraDist)
    distArraySr.append(distArrayS[i]+extraDist)

    #dot product
    dot = (v1x*v2x)+(v1y*v2y)+(v1z*v2z)

    sqrt1 = math.sqrt((v1x**2)+(v1y**2)+(v1z**2))
    sqrt2 = math.sqrt((v2x**2)+(v2y**2)+(v2z**2))
    try:
        angle = math.degrees(math.acos(dot/(sqrt1*sqrt2)))
    except:
        print dot,sqrt1,sqrt2,dot/(sqrt1*sqrt2)
    ele = 90-angle

    #sin(theta) = opp / hyp = altdiff / cartdist
    #so the elevation change with dist is... 
                        
    altDiff = math.sin(math.radians(ele))*cartDistance(x1,y1,z1,x2,y2,z2)
    #calc'd alt is alt2+h2+altDiff
    altArrayRounMod.append(h2+altDiff)
    #print altDiff
    
    
c = 299792458.0 #m/s    
    
size1 = 500
size2 = 5000
distStart = 4000#10000
distEnd = 19000#17000
d = np.linspace(distStart, distEnd, size2)
f = 1370*math.pow(10.0,6.0)
print f

#((y,x))
lamba = c/f
#print lamba
#print d
Pt = 1.0
Gt = 1.0
Gr = 1.0
altStart = 300
altEnd = 700
ht = np.linspace(altEnd, altStart, size1) #starts at the top of the image and makes pixels down... hence start at max alt and work to ground...
hr = h2
results = np.zeros((size1, size2))
"""
for i in range(len(ht)):
    for ii in range(len(d)):
        results[i,ii]= twoRayModel2(lamba,Pt, Gt, Gr, ht[i], hr, d[ii])
"""

#Lp1 = [freeSpaceLoss(lamba,Pt,Gt,Gr, f,x) for x in d]
#Lp2 = [twoRayModel3(Pt, Gt, Gr, ht, hr, x) for x in d]
#print criticalDist(lamba, ht, hr)

#resultsmax,resultsmin = results.max(), results.min()
#results = (results - resultsmin)/(resultsmax - resultsmin)
fig = plt.figure()#frameon=False)
fig.set_size_inches(float(distEnd-distStart)/200.0, float(altEnd-altStart)/200.0)
ax = fig.gca()
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)
#ax_2 = ax.twinx()
#ax_2.plot(distArrayS, rssi2Array)

ax.plot(distArrayS, altArray)
ax.plot(distArraySe, altArrayElipMod)
#ax.plot(distArrayS, zdiffe)
#ax.plot(distArrayS, zdiffr)
ax.plot(distArraySr, altArrayRounMod)
#ax.plot(distArrayS, distToAdde)
#ax.plot(distArrayS, distToAddr)  
ax.set_ylim(100,150)#altStart, altEnd)
#ax.imshow(results, cmap=plt.cm.gray, interpolation='none', extent=[distStart,distEnd,altStart,altEnd])
#ax_2.set_ylim(altStart, altEnd)
#fig1 = plt.gcf()
#fig1.savefig('tessstttyyy.png', dpi=200)


"""
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
ax.plot_wireframe(xa, ya, za, color="r")
"""
plt.show()
plt.close("all")
