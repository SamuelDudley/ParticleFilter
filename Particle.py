"""This class describes a particle"""
import Model

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
        self.RSS = 0
        self.kill = False
        self.error = 0
        
        

class Particle(object):
    def __init__(self):
        self.state  = PState()


        
        

        
    