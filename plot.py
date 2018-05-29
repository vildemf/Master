from numpy import *
from matplotlib.pyplot import *

filename = 'trainingvaluesADAM6hidden500epochs.txt'

f = open(filename, 'r')
Eloc = []
variance = []
gradientnorm = []

for line in f:
	line = line.split()
	Eloc.append(float(line[0]))
	variance.append(float(line[1]))
	gradientnorm.append(float(line[1]))




Eloc = array(Eloc)
av = Eloc[330:]
print sum(av)/av.size

plot(Eloc)
xlabel('Optimization iteration')
ylabel('Local energy')
title('nx=4 nh=2 gaussInit nSamples=10**4 metStep=2.5 sigma=omega=1 \n eta=0.01 interaction accept=0.75 oneCoordPrSampl')
show()