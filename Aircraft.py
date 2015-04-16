class AState(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        
        self.airSpeed = -10
        self.groundSpeed = -80
        
        self.RSS = 0


class Aircraft(object):
    def __init__(self):
        self.state = AState()