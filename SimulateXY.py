"""this file is part of a particle filter simulation"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import bisect
import time

import Model
import Particle as par
import Transmitter as tra
import Aircraft as air

class WeightedDistribution(object):
    def __init__(self, state):
        accum = 0.0
        self.state = [particle for particle in state if particle.state.error > 0]
        self.distribution = []
        for x in self.state:
            accum += x.state.error
            self.distribution.append(accum)
        
        print accum

    def pick(self):
        try:
            return self.state[bisect.bisect_left(self.distribution, random.uniform(0, 1))]
        except IndexError:
            # Happens when all particles are improbable w=0
            return None
        

class Map(object):
    """this class contains the simulation world"""
    def __init__(self):
        self.modelHelperFunctions = Model
        self.timeDelta = 1.0 #sec
        self.numberParticles = 1000
        
        self.world = [0,1500, #x dims
                      0,1500, #y dims
                      0,200] #z dims
        
        aircraft = air.Aircraft()
        aircraft.state.x = random.uniform(1000, 1500) #randomise the x location of the aircraft
        aircraft.state.y = random.uniform(1000, 1500) #randomise the y location of the aircraft
        aircraft.state.z = random.uniform(50, 120) #randomise the z location of the aircraft
        self.aircrafts = [aircraft]
        
        transmitter = tra.Transmitter()
        transmitter.state.x = 1 #set the x location of the transmitter
        transmitter.state.y = 1 #set the x location of the transmitter
        transmitter.state.z = 2.35 #the z location of the transmitter [meters off the ground]
        self.transmitters = [transmitter]
        
        
        self.particles = []
        for i in range(self.numberParticles):
            self.particles.append(par.Particle())
            self.particles[-1].state.x = random.uniform(self.world[0], self.world[1]) #randomise the x location of the particle
            self.particles[-1].state.y = random.uniform(self.world[2], self.world[3]) #randomise the y location of the particle
            self.particles[-1].state.z = random.uniform(self.world[4], self.world[5]) #randomise the z location of the particle
            #self.particles[-1].state.groundSpeed = random.uniform(-10, -100) #randomise ground speed of the particle, heading is assumed to be known
            self.particles[-1].state.groundSpeed = self.aircrafts[0].state.groundSpeed #randomise ground speed of the particle, heading is assumed to be known
            
    def step(self):
        #move the plane
        pass
        
        
    def update(self):
        for particle in self.particles:
            #calculate the RSS for the particles
            distance = self.modelHelperFunctions.cart_dist2(self.transmitters[0].state.x, self.transmitters[0].state.y, particle.state.x, particle.state.y)
            frequency = self.transmitters[0].state.frequency
            transmitterHeight = self.transmitters[0].state.z
            transmitterPower = self.transmitters[0].state.power
            receiverHeight = self.particles[-1].state.z
            
            particle.state.RSS = self.modelHelperFunctions.twoRay(distance, frequency, transmitterHeight, receiverHeight, transmitterPower)+ np.random.normal(0, 2., 1)[0] #randomise the x location of the particle
            
        #calculate the RSS for the aircraft
        distance = self.modelHelperFunctions.cart_dist2(self.transmitters[0].state.x, self.transmitters[0].state.y, self.aircrafts[0].state.x, self.aircrafts[0].state.y)
        frequency = self.transmitters[0].state.frequency
        transmitterHeight = self.transmitters[0].state.z
        transmitterPower = self.transmitters[0].state.power
        receiverHeight = self.aircrafts[0].state.z
        
        self.aircrafts[0].state.RSS = self.modelHelperFunctions.twoRay(distance, frequency, transmitterHeight, receiverHeight, transmitterPower)
        
        self.assess()
        
    def assess(self):

        for particle in self.particles:
            # values near to aircraft RSS measurement => 1, further away => 0
            particle.state.error = abs(particle.state.RSS-self.aircrafts[0].state.RSS) #calcuate the diffrence in RSS
            
            sigma2 = 3 ** 2
            particle.state.error= math.e ** - (particle.state.error ** 2 / (2 * sigma2))
            

        # Normalise weights
        nu = sum(particle.state.error for particle in self.particles)
        if nu:
            for particle in self.particles:
                particle.state.error /= nu
                #print particle.state.error
                
            
        
        # create a weighted distribution, for fast picking
        dist = WeightedDistribution(self.particles)
        
        new_particles =[]
        new = 0
        old = 0
        while len(new_particles) < self.numberParticles:
            p = dist.pick()
            new_p = par.Particle()
            if p is None:  # No pick b/c all totally improbable
                new_p.state.x = random.uniform(self.world[0], self.world[1]) #randomise the x location of the particle
                new_p.state.y = random.uniform(self.world[2], self.world[3]) #randomise the y location of the particle
                new_p.state.z = random.uniform(self.world[4], self.world[5]) #randomise the z location of the particle
                new_p.state.yaw = self.aircrafts[0].state.yaw + np.random.normal(0, 2., 1)[0]
                new_p.state.groundSpeed = random.uniform(10, 100) #randomise ground speed of the particle, heading is assumed to be known
                new+=1

            else:
                if len(new_particles)>0.8*self.numberParticles:
                    mod = 10
                else:
                    mod = 1
                new_p.state.x = p.state.x + np.random.normal(0, mod*2., 1)[0] #randomise the x location of the particle
                new_p.state.y = p.state.y + np.random.normal(0, mod*2., 1)[0] #randomise the y location of the particle
                new_p.state.z = p.state.z + np.random.normal(0, mod*2., 1)[0] #randomise the z location of the particle
                new_p.state.yaw = self.aircrafts[0].state.yaw + np.random.normal(0, 2., 1)[0]
                new_p.state.groundSpeed = p.state.groundSpeed + np.random.normal(0, mod*5., 1)[0] #randomise ground speed of the particle, heading is assumed to be known
                old+=1
            new_particles.append(new_p)

        self.particles = new_particles
        print old,new


            
    def step(self):
        #move the aircraft
        self.aircrafts[0].move(self.timeDelta)
        
        #move the particles
        for i in range(len(self.particles)):
            self.particles[i].move(self.timeDelta)
        
            
    
    def draw(self):
        
        #plt.ion() #non blocking plt.show()
        plt.figure()
        for particle in self.particles:
            if particle.state.kill == False: 
                plt.scatter(particle.state.x, particle.state.y, s=20, c='yellow')
            else:
                plt.scatter(particle.state.x, particle.state.y, s=10, c='yellow')
            #ax.arrow(particle.state.x,  particle.state.z, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
        for aircraft in self.aircrafts:
            plt.scatter(aircraft.state.x, aircraft.state.y, s=40, c='blue')
        
        plt.show()

        

        

test = Map()
test.draw()
test.update()
count = 0
while count < 2:
    test.step()
    test.update()
    test.draw()
    count += 1
        
    