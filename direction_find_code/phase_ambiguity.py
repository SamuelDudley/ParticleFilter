# we need to do either:
#have a material between the antenna to determine rough aoa
#have more than 2 antenna and reduce / contrain the ambiguity

#for the case of more than 2 antenna, this can be
#achived by using more than 1 mimo usrp b210
import numpy as np
import matplotlib.pyplot as plt

lamba = 0.23 #23cm ~1.3G
theta = np.pi/4.2
def num_ambiguities(s):
    #s is antenna baseline dist in m
    return int((s/lamba)*np.sin(theta))
    
    
a = 1.34
b = 1.35
c = 1.36

baseline1 = 1.0
baseline2 = 1.1

ant_12_phase = np.pi*a
ant_13_phase = np.pi*b

norm_ant_12 = ant_12_phase

amb1 =  num_ambiguities(baseline1)
amb2 =  num_ambiguities(baseline2)
print amb1, amb2
i1 = np.linspace(-amb1, amb1, 2*amb1 +1, endpoint = True)
i2 = np.linspace(-amb2, amb2, 2*amb2 +1, endpoint = True)

norm_phase = np.linspace(0,1,2, endpoint = True)
output = []
output2 = []

for i1_ in i1:
    for i2_ in i2:
        output.append(eval('(baseline1/baseline2)*norm_phase +((baseline1/baseline2)*i1_-i2_)'))



plt.figure()
for out in output:
    plt.plot(norm_phase,out)
plt.axis([0,1,
          0,1])
plt.show()

print norm_ant_12

pass