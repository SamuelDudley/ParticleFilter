
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
    
    #print r1, r2
    
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

def cart_dist3(x1,y1,z1,x2,y2,z2):
    return math.sqrt(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))

def cart_dist2(x1,y1,x2,y2):
    return math.sqrt(((x2-x1)**2)+((y2-y1)**2))

def x_dist(x1,x2):
    return (x2-x1)
    
    

def vinc_dist(  f,  a,  phi1,  lembda1,  phi2,  lembda2 ) :
    """ 
    Returns the distance between two geographic points on the ellipsoid
    and the forward and reverse azimuths between these points.
    lats, longs and azimuths are in decimal degrees, distance in metres 

    Returns ( s, alpha12,  alpha21 ) as a tuple
    """
    f = 1.0 / 298.257223563
    a = 6378137.0  
    
    phi1 = math.degrees(phi1)
    phi2 = math.degrees(phi2)
    lembda1 = math.degrees(lembda1)
    lembda2 = math.degrees(lembda2)
    
    
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