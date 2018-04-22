from numpy import *
from matplotlib.pyplot import *

filename = 'RBMoutput.txt'

f = open(filename, 'r')
Eloc = []
variance = []

for line in f:
	line = line.split()
	Eloc.append(line[1])
	variance.append(line[2])


Eloc = array(Eloc)

plot(Eloc)
xlabel('Optimization iteration')
ylabel('Local energy')
title('nx=4 nh=2 gaussInit nSamples=10**4 metStep=2.5 sigma=omega=1 \n eta=0.01 interaction accept=0.75 oneCoordPrSampl')
show()