"""This class describes a particle"""
import numpy as np

class PState(object):
    def __init__(self, transmitterCount):
        self.x = 0
        self.y = 0
        self.z = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.airSpeed = 0
        self.groundSpeed = 0
        
        
        self.RSS = np.zeros(transmitterCount)
        self.angle = np.zeros(transmitterCount)
        self.RSSError = 0
        self.angleError = 0
        self.error = 0
        self.selected = False
        
        

class Particle(object):
    def __init__(self, transmitterCount):
        self.state  = PState(transmitterCount)

        
    def move(self, timeDelta):
        self.state.x += timeDelta*(self.state.groundSpeed*np.sin(np.deg2rad(self.state.yaw)))
        self.state.y += timeDelta*(self.state.groundSpeed*np.cos(np.deg2rad(self.state.yaw)))
        
    