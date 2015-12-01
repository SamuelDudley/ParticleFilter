import numpy as np
class AState(object):
    def __init__(self, transmitterCount):
        self.x = 0
        self.y = 0
        self.z = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 25.

        self.airSpeed = -10
        self.groundSpeed = 20.
        
        self.RSS = np.zeros(transmitterCount)
        self.angle = np.zeros(transmitterCount)
        self.RSSError = 0
        self.angleError = 0
        self.error = 0


class Aircraft(object):
    def __init__(self, transmitterCount):
        self.state = AState(transmitterCount)
        
    def update_location(self):
        new_location = self.get_location_from_socket()
        self.state.x = 
    
    def move(self, timeDelta):
        self.state.x += timeDelta*(self.state.groundSpeed*np.sin(np.deg2rad(self.state.yaw)))
        self.state.y += timeDelta*(self.state.groundSpeed*np.cos(np.deg2rad(self.state.yaw)))