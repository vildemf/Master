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
title('RBM energy optimization')
show()