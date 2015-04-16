
class TState(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        
        self.power = 2000
        self.frequency = 1380 #MHz
        
        #antenna pattern
        #moving?


class Transmitter(object):
    def __init__(self):
        self.state = TState()
