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
import Confidence as con

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
        random.seed()
        self.modelHelperFunctions = Model
        self.timeDelta = 1.0 #sec
        self.numberParticles = 1000
        
        self.world = np.array([-3000,3000, #x dims
                      -3000,3000, #y dims
                      0,200]) #z dims
        
       
        self.transmitters = np.array([tra.Transmitter()])
        
        self.transmitters[-1].state.x = 0 #set the x location of the transmitter
        self.transmitters[-1].state.y = 0 #set the x location of the transmitter
        self.transmitters[-1].state.z = 2.35 #the z location of the transmitter [meters off the ground]
        self.transmitters[-1].state.id = len(self.transmitters)-1
          
        self.transmitters = np.hstack((self.transmitters,np.array([tra.Transmitter()])))
        self.transmitters[-1].state.x = 500 #set the x location of the transmitter
        self.transmitters[-1].state.y = -100 #set the x location of the transmitter
        self.transmitters[-1].state.z = 4. #the z location of the transmitter [meters off the ground]
        self.transmitters[-1].state.id = len(self.transmitters)-1 
         
        self.transmitters = np.hstack((self.transmitters,np.array([tra.Transmitter()])))
        self.transmitters[-1].state.x = 1000 #set the x location of the transmitter
        self.transmitters[-1].state.y = 1500 #set the x location of the transmitter
        self.transmitters[-1].state.z = 3. #the z location of the transmitter [meters off the ground]
        self.transmitters[-1].state.id = len(self.transmitters)-1
         
        


        self.aircrafts = np.array([air.Aircraft(len(self.transmitters))])
        for aircraft in self.aircrafts:
            aircraft.state.x = -100#random.uniform(300, 900) #randomise the x location of the aircraft
            aircraft.state.y = 1400#random.uniform(300, 900) #randomise the y location of the aircraft
            aircraft.state.z = random.uniform(50, 120) #randomise the z location of the aircraft
        
        
        self.particles = []
        for i in range(self.numberParticles):
            self.particles.append(par.Particle(len(self.transmitters)))
            self.particles[-1].state.x = random.uniform(self.world[0], self.world[1]) #randomise the x location of the particle
            self.particles[-1].state.y = random.uniform(self.world[2], self.world[3]) #randomise the y location of the particle
            self.particles[-1].state.z = random.uniform(self.world[4], self.world[5]) #randomise the z location of the particle
            self.particles[-1].state.groundSpeed = self.aircrafts[0].state.groundSpeed #randomise ground speed of the particle, heading is assumed to be known
            self.particles[-1].state.yaw = self.aircrafts[0].state.yaw #randomise ground speed of the particle, heading is assumed to be known
        
        
        
    def update(self):
        for particle in self.particles:
            #calculate the RSS for the particles
            for transmitter in self.transmitters:
                distance = self.modelHelperFunctions.cart_dist2(transmitter.state.x, transmitter.state.y, particle.state.x, particle.state.y)
                frequency = transmitter.state.frequency
                transmitterHeight = transmitter.state.z
                transmitterPower = transmitter.state.power
                receiverHeight = particle.state.z
                
                particle.state.RSS[transmitter.state.id] = self.modelHelperFunctions.twoRay(distance, frequency, transmitterHeight, receiverHeight, transmitterPower)+ np.random.normal(0, 2., 1)[0] #randomise the x location of the particle
                particle.state.angle[transmitter.state.id] = self.modelHelperFunctions.angle(transmitter.state.x, transmitter.state.y, particle.state.x, particle.state.y) + np.random.normal(0, 10., 1)[0]
                
        for aircraft in self.aircrafts:    
            #calculate the RSS for the aircraft
            for transmitter in self.transmitters:
                distance = self.modelHelperFunctions.cart_dist2(transmitter.state.x, transmitter.state.y, self.aircrafts[0].state.x, self.aircrafts[0].state.y)
                frequency = transmitter.state.frequency
                transmitterHeight = transmitter.state.z
                transmitterPower = transmitter.state.power
                receiverHeight = aircraft.state.z
                
                aircraft.state.RSS[transmitter.state.id] = self.modelHelperFunctions.twoRay(distance, frequency, transmitterHeight, receiverHeight, transmitterPower)
                aircraft.state.angle[transmitter.state.id] = self.modelHelperFunctions.angle(transmitter.state.x, transmitter.state.y, self.aircrafts[0].state.x, self.aircrafts[0].state.y) + np.random.normal(0, 10., 1)[0]
                
        self.assess()
        
    def boot_strap(self):
        new_particles =[]
        for particle in self.particles:
            new_p = par.Particle(len(self.transmitters))
            if len(new_particles)>0.5*self.numberParticles:
                mod = 10
            else:
                mod = 1
            new_p.state.x =self.aircrafts[0].state.x + np.random.normal(0, mod*2., 1)[0] #randomise the x location of the particle
            new_p.state.y = self.aircrafts[0].state.y + np.random.normal(0, mod*2., 1)[0] #randomise the y location of the particle
            new_p.state.z = self.aircrafts[0].state.z + np.random.normal(0, mod*2., 1)[0] #randomise the z location of the particle
            new_p.state.yaw = self.aircrafts[0].state.yaw + np.random.normal(0, 2., 1)[0]
            new_p.state.groundSpeed = self.aircrafts[0].state.groundSpeed + np.random.normal(0, mod*5., 1)[0] #randomise ground speed of the particle, heading is assumed to be known

            new_particles.append(new_p)

        self.particles = new_particles
            
        
    def assess(self):
        angleWeight = 1
        RSSWeight = 1
        for particle in self.particles:
            # values near to aircraft RSS measurement => 1, further away => 0
            for RSSIIndex in range(len(particle.state.RSS)):
                particle.state.RSSError += abs(particle.state.RSS[RSSIIndex]-self.aircrafts[0].state.RSS[RSSIIndex]) #calcuate the diffrence in RSS
            #particle.state.error = particle.state.error/len(particle.state.RSS)
            for angleIndex in range(len(particle.state.angle)):
                particle.state.angleError += abs(self.modelHelperFunctions.comp_angle_deg(particle.state.angle[angleIndex],self.aircrafts[0].state.angle[angleIndex]))
                
            particle.state.error =  (angleWeight*particle.state.angleError) +(RSSWeight*particle.state.RSSError)
                
            sigma2 = 20. ** 2
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
        old_mod = 0
        old = 0
        while len(new_particles) < self.numberParticles:
            p = dist.pick()
            new_p = par.Particle(len(self.transmitters))
            if p is None:  # No pick b/c all totally improbable
                new_p.state.x = random.uniform(self.world[0], self.world[1]) #randomise the x location of the particle
                new_p.state.y = random.uniform(self.world[2], self.world[3]) #randomise the y location of the particle
                new_p.state.z = random.uniform(self.world[4], self.world[5]) #randomise the z location of the particle
                new_p.state.yaw = self.aircrafts[0].state.yaw + np.random.normal(0, 2., 1)[0]
                new_p.state.groundSpeed = random.uniform(10, 100) #randomise ground speed of the particle, heading is assumed to be known
                new+=1

            else:
                if not p.state.selected:
                    p.state.selected = True
                    new_p.state.x = p.state.x 
                    new_p.state.y = p.state.y
                    new_p.state.z = p.state.z
                    new_p.state.yaw = self.aircrafts[0].state.yaw + np.random.normal(0, 2., 1)[0]
                    new_p.state.groundSpeed = p.state.groundSpeed
                    old+=1
                else:    
                    if len(new_particles)>0.9*self.numberParticles:
                        mod = 10
                    else:
                        mod = 1
                    new_p.state.x = p.state.x + np.random.normal(0, mod*5., 1)[0] #randomise the x location of the particle
                    new_p.state.y = p.state.y + np.random.normal(0, mod*5., 1)[0] #randomise the y location of the particle
                    new_p.state.z = p.state.z + np.random.normal(0, mod*5., 1)[0] #randomise the z location of the particle
                    new_p.state.yaw = self.aircrafts[0].state.yaw + np.random.normal(0, 2., 1)[0]
                    new_p.state.groundSpeed = p.state.groundSpeed + np.random.normal(0, mod*2., 1)[0] #randomise ground speed of the particle, heading is assumed to be known
                    old_mod+=1
            new_particles.append(new_p)

        self.particles = new_particles
        print old,old_mod,new


            
    def step(self):
        #move the aircraft
        self.aircrafts[0].move(self.timeDelta)
        
        #move the particles
        for i in range(len(self.particles)):
            self.particles[i].move(self.timeDelta)
        
            
    
    def draw(self):
        
        #plt.ion() #non blocking plt.show()
        plt.figure()
        x = [particle.state.x for particle in self.particles]
        y = [particle.state.y for particle in self.particles]
        points = np.zeros((len(self.particles), 2))
        points[:,0] = x
        points[:,1] = y

        mean= np.mean(points, 0)

            #ax.arrow(particle.state.x,  particle.state.z, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
        for aircraft in self.aircrafts:
            plt.scatter(aircraft.state.x, aircraft.state.y, s=40, c='blue')
            
        for transmitter in self.transmitters:
            plt.scatter(transmitter.state.x, transmitter.state.y, s=40, c='red')
        
        plt.scatter(x, y, s=20, c='yellow') 
        plt.scatter(mean[0],mean[1], s=80, c='red')
        
        con.plot_point_cov(points, nstd=3, alpha=0.5, color='green')
        plt.show()

        

        

test = Map()
#test.draw()
#test.boot_strap()
test.update()
count = 0
while count < 200:
    test.step()
    test.update()
    #test.draw()
    count += 1
test.draw()
    