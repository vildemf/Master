from numpy import *
from matplotlib.pyplot import *



def readfile(filename):
	f = open(filename, 'r')
	Eloc = []
	variance = []
	gradientnorm = []

	for line in f:
		line = line.split()
		Eloc.append(float(line[0]))
		variance.append(float(line[1]))
		gradientnorm.append(float(line[1]))
	return Eloc, gradientnorm

#filenames = ['trainingvaluesAdam005.txt', 'trainingvaluesAdam05.txt', 'trainingvaluesAdam5.txt']
filenames = ['trainingvaluesInt2hiddenADAM.txt', 'trainingvaluesInt4hiddenADAM.txt', 'trainingvaluesInt6hiddenADAM.txt']
#filenames = ['trainingvaluesInt2hidden.txt', 'trainingvaluesInt4hidden.txt', 'trainingvaluesInt6hidden.txt']
#filenames = ['trainingvalues005.txt', 'trainingvalues05.txt', 'trainingvalues5.txt']

Elocs = []
gradnorms = []
for filename in filenames:
	Eloc, gradnorm = readfile(filename)
	Elocs.append(array(Eloc))
	gradnorms.append(array(gradnorm))


# Simple Eloc vs epochs plot
"""
Eloc = array(Eloc)
av = Eloc[330:]
print sum(av)/av.size

plot(log(abs(Eloc-2.0)))
xlabel('Optimization iteration')
ylabel('Local energy')
title('nx=4 nh=2 gaussInit nSamples=10**4 metStep=2.5 sigma=omega=1 \n eta=0.01 interaction accept=0.75 oneCoordPrSampl')
show()
"""


# Eloc and grad norm vs epochs plot

Eloc005 = log(abs(Elocs[0] - 3.0))
Eloc05  = log(abs(Elocs[1] - 3.0))
Eloc5   = log(abs(Elocs[2] - 3.0))

gradnorms005 = log(abs(gradnorms[0]))
gradnorms05  = log(abs(gradnorms[1]))
gradnorms5   = log(abs(gradnorms[2]))

f, ((ax01, ax02, ax03), (ax11, ax12, ax13), (ax21, ax22, ax23)) = subplots(3, 3, sharex='col', sharey='row')
ax01.plot(Elocs[0], color='#D62728')
ax02.plot(Elocs[1], color='#1F77B4')
ax03.plot(Elocs[2], color='#17BECF')
ax01.set_ylabel("$E$ [a.u.]")
ax01.set_ylim(2, 4)

ax01.set_title('$H=2$')
ax02.set_title('$H=4$')
ax03.set_title('$H=6$')

ax11.plot(Eloc005, color='#D62728')     #'#7b2cb3')
ax12.plot(Eloc05, color='#1F77B4')              #'#4d7fe2')
ax13.plot(Eloc5, color='#17BECF')    #'#272a9d')
ax11.set_ylabel("log($|\Delta E|$)", fontsize=15)

ax21.plot(gradnorms005, color='#D62728')
ax22.plot(gradnorms05, color='#1F77B4')        #'#2CA02C')
ax23.plot(gradnorms5, color='#17BECF')

ax21.set_ylabel("log($| \\nabla_\\alpha E|$)", fontsize=15)
ax22.set_xlabel("Gradient descent iteration", fontsize=15)
ax21.set_xticks([0, 200, 400, 600, 800])
ax22.set_xticks([0, 200, 400, 600, 800])
ax23.set_xticks([0, 200, 400, 600, 800])
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
#f.subplots_adjust(hspace=0)
#setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#f.suptitle('Learning with different learning rates', fontsize=14)
show()
