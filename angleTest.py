import numpy as np
def angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    rads = np.arctan2(dy,dx)
        
    return np.degrees(rads)

def comp_angle_deg(angle1, angle2):
    deg = angle1-angle2
    print deg
    if deg < 0: #negitive
        deg += 360.
        
    if abs(deg) > 180.:
        deg = 360. - abs(deg)
        
    return deg 


angle1 = 90
angle2 = -45

print comp_angle_deg(angle1,angle2)
