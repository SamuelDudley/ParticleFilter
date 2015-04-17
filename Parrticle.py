"""This class describes a particle"""

class PState(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.airSpeed = 0
        self.groundSpeed = 0
        
        

class Particle(object):
    def __init__(self):
        self.state  = PState
        
    def move(self, timeDelta):
        self.state.x += timeDelta*(self.state.groundSpeed*np.sin(np.deg2rad(self.state.yaw)))
        self.state.y += timeDelta*(self.state.groundSpeed*np.cos(np.deg2rad(self.state.yaw)))
        
    