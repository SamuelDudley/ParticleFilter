
import math
import cmath
import numpy as np

def normalize_length(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm

def normalize_values(v):
    return v/(np.max(v))

def twoRay(distance, frequency, transmitterHeight, receiverHeight, transmitterPower, transmitterGain =1, receiverGain=1):
#     print ""
#     print distance, frequency, transmitterPower, transmitterGain, receiverGain, transmitterHeight, receiverHeight
#     print ""
    #distance [m], frequency [MHz]
    c = 299792458.0 #speed of light m/s
    f = frequency*math.pow(10.0,6.0) #convert from MHz to Hz

    lamba = c/f #wavelength in m
    
    R = -0.8 #reflection loss
    
    #Radio propergation for modern wireless systems henry l. bertoni
    r1=((distance**2.)+((transmitterHeight-receiverHeight)**2.))**(.5)
    r2=((distance**2.)+((transmitterHeight+receiverHeight)**2.))**(.5)
    
    print r1, r2
    
    beta = (2*math.pi)/lamba
    
    
    #Pr =((lamba/(4*math.pi*d))**2)*abs(1+R*cmath.exp(1j*beta*((2*transmitterHeight*receiverHeight)/distance)))**2
    #Pr = transmitterPower*transmitterGain*receiverGain*2*((lamba/(4*math.pi*distance))**2)*(1-math.cos((2*beta*transmitterHeight*receiverHeight)/distance))
    #Pr = transmitterPower*transmitterGain*receiverGain*2*((lamba/(4*math.pi*distance))**2)*abs((math.sin((beta*transmitterHeight*receiverHeight)/distance)))**2
    #Pr = transmitterPower*transmitterGain*receiverGain*((lamba/(4*cmath.pi*distance))**2)*abs(1+R*cmath.exp(1j*beta*((2*transmitterHeight*receiverHeight)/distance)))**2
    Pr = transmitterPower*transmitterGain*receiverGain*((lamba/(4*cmath.pi))**2)*abs(
                    ((cmath.exp(1j*beta*r1))/r1)+(R*(cmath.exp(1j*beta*r2))/r2))**2
    #print Pr
    Pr = 10*np.log10(Pr)
    return Pr

def freeSpace(distance,frequency, gain=15, exponent = 2):
    #distance [m], frequency [MHz]
    freeSpaceLoss = -32.44-20*math.log10(frequency)-20*math.log10(distance/1000.)
    return freeSpaceLoss+gain

def cart_dist3(x1,y1,z1,x2,y2,z2):
    return math.sqrt(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))

def cart_dist2(x1,y1,x2,y2):
    return math.sqrt(((x2-x1)**2)+((y2-y1)**2))

def x_dist(x1,x2):
    return (x2-x1)

def angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    rads = np.arctan2(dy,dx)
    return np.degrees(rads)


def comp_angle_rads(angle1, angle2):
    rads = angle1-angle2
    if rads < 0: #negitive
        rads += 2*np.pi
        
    if abs(rads) > np.pi:
        rads = 2*np.pi - abs(rads)
        
    return rads
        
def comp_angle_deg(angle1, angle2):
    deg = angle1-angle2
    if deg < 0: #negitive
        deg += 360.
        
    if abs(deg) > 180.:
        deg = 360. - abs(deg)
        
    return deg  


# --------------------------------------------------------------------- 
# |                                                                     |
# |    geodetic.cc -  a collection of geodetic functions                   |
# |    Jim Leven  - Dec 99                                                 |
# |                                                                     |
# | originally from:                                                    |
# | http://wegener.mechanik.tu-darmstadt.de/GMT-Help/Archiv/att-8710/Geodetic_py |                                                                   |
# |                                                                     |
# --------------------------------------------------------------------- 
# 
# 
# ----------------------------------------------------------------------
# | Algrothims from Geocentric Datum of Australia Technical Manual        |
# |                                                                     |
# | http://www.anzlic.org.au/icsm/gdatum/chapter4.html                    |
# |                                                                     |
# | This page last updated 11 May 1999                                     |
# |                                                                     |
# | Computations on the Ellipsoid                                        |
# |                                                                     |
# | There are a number of formulae that are available                   |
# | to calculate accurate geodetic positions,                             |
# | azimuths and distances on the ellipsoid.                            |
# |                                                                     |
# | Vincenty's formulae (Vincenty, 1975) may be used                     |
# | for lines ranging from a few cm to nearly 20,000 km,                 |
# | with millimetre accuracy.                                             |
# | The formulae have been extensively tested                             |
# | for the Australian region, by comparison with results               |
# | from other formulae (Rainsford, 1955 & Sodano, 1965).                 |
# |                                                                        |
# | * Inverse problem: azimuth and distance from known                     |
# |            latitudes and longitudes                                     |
# | * Direct problem: Latitude and longitude from known                 |
# |            position, azimuth and distance.                             |
# | * Sample data                                                         |
# | * Excel spreadsheet                                                 |
# |                                                                     |
# | Vincenty's Inverse formulae                                            |
# | Given: latitude and longitude of two points                         |
# |            (phi1, lembda1 and phi2, lembda2),                             |
# | Calculate: the ellipsoidal distance (s) and                         |
# | forward and reverse azimuths between the points (alpha12, alpha21).    |
# |                                                                        |
# ---------------------------------------------------------------------- 

import math


def vinc_dist(  f,  a,  phi1,  lembda1,  phi2,  lembda2 ) :
    """ 
    Returns the distance between two geographic points on the ellipsoid
    and the forward and reverse azimuths between these points.
    lats, longs and azimuths are in decimal degrees, distance in metres 
    
    Returns ( s, alpha12,  alpha21 ) as a tuple
    """
    
    if (abs( phi2 - phi1 ) < 1e-8) and ( abs( lembda2 - lembda1) < 1e-8 ) :
            return 0.0, 0.0, 0.0
    
    piD4   = math.atan( 1.0 )
    two_pi = piD4 * 8.0
    
    phi1    = phi1 * piD4 / 45.0
    lembda1 = lembda1 * piD4 / 45.0        # unfortunately lambda is a key word!
    phi2    = phi2 * piD4 / 45.0
    lembda2 = lembda2 * piD4 / 45.0
    
    b = a * (1.0 - f)
    
    TanU1 = (1-f) * math.tan( phi1 )
    TanU2 = (1-f) * math.tan( phi2 )
    
    U1 = math.atan(TanU1)
    U2 = math.atan(TanU2)
    
    lembda = lembda2 - lembda1
    last_lembda = -4000000.0        # an impossibe value
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
    return s, alpha12,  alpha21 

# END of Vincenty's Inverse formulae 


#-------------------------------------------------------------------------------
# Vincenty's Direct formulae                            |
# Given: latitude and longitude of a point (phi1, lembda1) and             |
# the geodetic azimuth (alpha12)                         |
# and ellipsoidal distance in metres (s) to a second point,            |
#                                         |
# Calculate: the latitude and longitude of the second point (phi2, lembda2)     |
# and the reverse azimuth (alpha21).                        |
#                                         |
#-------------------------------------------------------------------------------

def  vinc_pt( f, a, phi1, lembda1, alpha12, s ) :
    """
    
    Returns the lat and long of projected point and reverse azimuth
    given a reference point and a distance and azimuth to project.
    lats, longs and azimuths are passed in decimal degrees
    
    Returns ( phi2,  lambda2,  alpha21 ) as a tuple 
    
    """
    
    piD4 = math.atan( 1.0 )
    two_pi = piD4 * 8.0
    
    phi1    = phi1    * piD4 / 45.0
    lembda1 = lembda1 * piD4 / 45.0
    alpha12 = alpha12 * piD4 / 45.0
    if ( alpha12 < 0.0 ) : 
            alpha12 = alpha12 + two_pi
    if ( alpha12 > two_pi ) : 
            alpha12 = alpha12 - two_pi
    
    b = a * (1.0 - f)
    
    TanU1 = (1-f) * math.tan(phi1)
    U1 = math.atan( TanU1 )
    sigma1 = math.atan2( TanU1, math.cos(alpha12) )
    Sinalpha = math.cos(U1) * math.sin(alpha12)
    cosalpha_sq = 1.0 - Sinalpha * Sinalpha
    
    u2 = cosalpha_sq * (a * a - b * b ) / (b * b)
    A = 1.0 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * \
            (320 - 175 * u2) ) )
    B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2) ) )
    
    # Starting with the approximation
    sigma = (s / (b * A))
    
    last_sigma = 2.0 * sigma + 2.0    # something impossible
    
    # Iterate the following three equations 
    #  until there is no significant change in sigma 
    
    # two_sigma_m , delta_sigma
    while ( abs( (last_sigma - sigma) / sigma) > 1.0e-9 ) :
        two_sigma_m = 2 * sigma1 + sigma

        delta_sigma = B * math.sin(sigma) * ( math.cos(two_sigma_m) \
                + (B/4) * (math.cos(sigma) * \
                (-1 + 2 * math.pow( math.cos(two_sigma_m), 2 ) -  \
                (B/6) * math.cos(two_sigma_m) * \
                (-3 + 4 * math.pow(math.sin(sigma), 2 )) *  \
                (-3 + 4 * math.pow( math.cos (two_sigma_m), 2 ))))) \

        last_sigma = sigma
        sigma = (s / (b * A)) + delta_sigma
    
    phi2 = math.atan2 ( (math.sin(U1) * math.cos(sigma) + math.cos(U1) * math.sin(sigma) * math.cos(alpha12) ), \
            ((1-f) * math.sqrt( math.pow(Sinalpha, 2) +  \
            pow(math.sin(U1) * math.sin(sigma) - math.cos(U1) * math.cos(sigma) * math.cos(alpha12), 2))))
    
    lembda = math.atan2( (math.sin(sigma) * math.sin(alpha12 )), (math.cos(U1) * math.cos(sigma) -  \
            math.sin(U1) *  math.sin(sigma) * math.cos(alpha12)))
    
    C = (f/16) * cosalpha_sq * (4 + f * (4 - 3 * cosalpha_sq ))
    
    omega = lembda - (1-C) * f * Sinalpha *  \
            (sigma + C * math.sin(sigma) * (math.cos(two_sigma_m) + \
            C * math.cos(sigma) * (-1 + 2 * math.pow(math.cos(two_sigma_m),2) )))
    
    lembda2 = lembda1 + omega
    
    alpha21 = math.atan2 ( Sinalpha, (-math.sin(U1) * math.sin(sigma) +  \
            math.cos(U1) * math.cos(sigma) * math.cos(alpha12)))
    
    alpha21 = alpha21 + two_pi / 2.0
    if ( alpha21 < 0.0 ) :
            alpha21 = alpha21 + two_pi
    if ( alpha21 > two_pi ) :
            alpha21 = alpha21 - two_pi
    
    phi2       = phi2       * 45.0 / piD4
    lembda2    = lembda2    * 45.0 / piD4
    alpha21    = alpha21    * 45.0 / piD4
    
    return phi2,  lembda2,  alpha21 
    
    # END of Vincenty's Direct formulae

#--------------------------------------------------------------------------
# Notes: 
# 
# * "The inverse formulae may give no solution over a line 
#     between two nearly antipodal points. This will occur when 
#     lembda ... is greater than pi in absolute value". (Vincenty, 1975)
#  
# * In Vincenty (1975) L is used for the difference in longitude, 
#     however for consistency with other formulae in this Manual, 
#     omega is used here. 
# 
# * Variables specific to Vincenty's formulae are shown below, 
#     others common throughout the manual are shown in the Glossary. 
# 
# 
# alpha = Azimuth of the geodesic at the equator
# U = Reduced latitude
# lembda = Difference in longitude on an auxiliary sphere (lembda1 & lembda2 
#         are the geodetic longitudes of points 1 & 2)
# sigma = Angular distance on a sphere, from point 1 to point 2
# sigma1 = Angular distance on a sphere, from the equator to point 1
# sigma2 = Angular distance on a sphere, from the equator to point 2
# sigma_m = Angular distance on a sphere, from the equator to the 
#         midpoint of the line from point 1 to point 2
# u, A, B, C = Internal variables
# 
# 
#
# 
#*******************************************************************

# Test driver

# if __name__ == "__main__" :
# 
#         f = 1.0 / 298.257223563        # WGS84
#         a = 6378137.0             # metres
# 
#         print  "\n Ellipsoidal major axis =  %12.3f metres\n" % ( a )
#         print  "\n Inverse flattening     =  %15.9f\n" % ( 1.0/f )
# 
#         print "\n Test Flinders Peak to Buninyon"
#         print "\n ****************************** \n"
#         phi1 = -(( 3.7203 / 60. + 57) / 60. + 37 )
#         lembda1 = ( 29.5244 / 60. + 25) / 60. + 144
#         print "\n Flinders Peak = %12.6f, %13.6f \n" % ( phi1, lembda1 )
#         deg = int(phi1)
#         min = int(abs( ( phi1 - deg) * 60.0 ))
#         sec = abs(phi1 * 3600 - deg * 3600) - min * 60
#         print " Flinders Peak =   %3i\xF8%3i\' %6.3f\",  " % ( deg, min, sec ),
#         deg = int(lembda1)
#         min = int(abs( ( lembda1 - deg) * 60.0 ))
#         sec = abs(lembda1 * 3600 - deg * 3600) - min * 60
#         print " %3i\xF8%3i\' %6.3f\" \n" % ( deg, min, sec )
# 
#         phi2 = -(( 10.1561 / 60. + 39) / 60. + 37 )
#         lembda2 = ( 35.3839 / 60. + 55) / 60. + 143
#         print "\n Buninyon      = %12.6f, %13.6f \n" % ( phi2, lembda2 )
# 
#         deg = int(phi2)
#         min = int(abs( ( phi2 - deg) * 60.0 ))
#         sec = abs(phi2 * 3600 - deg * 3600) - min * 60
#         print " Buninyon      =   %3i\xF8%3i\' %6.3f\",  " % ( deg, min, sec ),
#         deg = int(lembda2)
#         min = int(abs( ( lembda2 - deg) * 60.0 ))
#         sec = abs(lembda2 * 3600 - deg * 3600) - min * 60
#         print " %3i\xF8%3i\' %6.3f\" \n" % ( deg, min, sec )
# 
#         dist, alpha12, alpha21   = vinc_dist  ( f, a, phi1, lembda1, phi2,  lembda2 )
# 
#         print "\n Ellipsoidal Distance = %15.3f metres\n            should be         54972.271 m\n" % ( dist )
#         print "\n Forward and back azimuths = %15.6f, %15.6f \n" % ( alpha12, alpha21 )
#         deg = int(alpha12)
#         min = int( abs(( alpha12 - deg) * 60.0 ) )
#         sec = abs(alpha12 * 3600 - deg * 3600) - min * 60
#         print " Forward azimuth = %3i\xF8%3i\' %6.3f\"\n" % ( deg, min, sec )
#         deg = int(alpha21)
#         min = int(abs( ( alpha21 - deg) * 60.0 ))
#         sec = abs(alpha21 * 3600 - deg * 3600) - min * 60
#         print " Reverse azimuth = %3i\xF8%3i\' %6.3f\"\n" % ( deg, min, sec )
# 
# 
#         # Test the direct function */
#         phi1 = -(( 3.7203 / 60. + 57) / 60. + 37 )
#         lembda1 = ( 29.5244 / 60. + 25) / 60. + 144
#         dist = 54972.271
#         alpha12 = ( 5.37 / 60. + 52) / 60. + 306
#         phi2 = lembda2 = 0.0
#         alpha21 = 0.0
# 
#         phi2, lembda2, alpha21 = vinc_pt (  f, a, phi1, lembda1, alpha12, dist )
# 
#         print "\n Projected point =%11.6f, %13.6f \n" % ( phi2, lembda2 )
#         deg = int(phi2)
#         min = int(abs( ( phi2 - deg) * 60.0 ))
#         sec = abs( phi2 * 3600 - deg * 3600) - min * 60
#         print " Projected Point = %3i\xF8%3i\' %6.3f\", " % ( deg, min, sec ),
#         deg = int(lembda2)
#         min = int(abs( ( lembda2 - deg) * 60.0 ))
#         sec = abs(lembda2 * 3600 - deg * 3600) - min * 60
#         print "  %3i\xF8%3i\' %6.3f\"\n" % ( deg, min, sec )
#         print " Should be Buninyon \n" 
#         print "\n Reverse azimuth = %10.6f \n" % ( alpha21 )
#         deg = int(alpha21)
#         min = int(abs( ( alpha21 - deg) * 60.0 ))
#         sec = abs(alpha21 * 3600 - deg * 3600) - min * 60
#         print " Reverse azimuth = %3i\xF8%3i\' %6.3f\"\n\n" % ( deg, min, sec )

#*******************************************************************

#         f = 1.0 / 298.257223563        # WGS84
#         a = 6378137.0             # metres
def est_dist(  f,  a,  phi1,  lembda1,  phi2,  lembda2 ) :
    """ 

    Returns an estimate of the distance between two geographic points
    This is a quick and dirty vinc_dist 
    which will generally estimate the distance to within 1%
    Returns distance in metres

    """

    piD4   = 0.785398163397 

    phi1    = phi1 * piD4 / 45.0
    lembda1 = lembda1 * piD4 / 45.0
    phi2    = phi2 * piD4 / 45.0
    lembda2 = lembda2 * piD4 / 45.0

    c = math.cos((phi2+phi1)/2.0)

    return math.sqrt( pow(math.fabs(phi2-phi1), 2) + \
            pow(math.fabs(lembda2-lembda1)*c, 2) ) * a * ( 1.0 - f + f * c )

   # END of rough estimate of the distance.

