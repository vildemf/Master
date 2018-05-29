TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    trainer.cpp \
    montecarlomethod.cpp \
    quantummodel.cpp \
    sampler/sampler.cpp \
    hamiltonian.cpp \
    sampler/gibbs/gibbs.cpp \
    sampler/metropolis/metropolis.cpp \
    sampler/metropolis/metropolisbruteforce/metropolisbruteforce.cpp \
    sampler/metropolis/metropolishastings/metropolishastings.cpp \
    neuralquantumstate/neuralquantumstatepositivedefinite/neuralquantumstatepositivedefinite.cpp \
    neuralquantumstate/nerualquantumstate.cpp \
    gradientdescent/gradientdescent.cpp \
    gradientdescent/gradientdescentsimple/gradientdescentsimple.cpp \
    gradientdescent/gradientdescentadam/gradientdescentadam.cpp


INCLUDEPATH += /usr/local/include/eigen3/

HEADERS += \
    trainer.h \
    montecarlomethod.h \
    quantummodel.h \
    sampler/sampler.h \
    hamiltonian.h \
    sampler/gibbs/gibbs.h \
    sampler/metropolis/metropolis.h \
    sampler/metropolis/metropolisbruteforce/metropolisbruteforce.h \
    sampler/metropolis/metropolishastings/metropolishastings.h \
    neuralquantumstate/neuralquantumstatepositivedefinite/neuralquantumstatepositivedefinite.h \
    neuralquantumstate/nerualquantumstate.h \
    gradientdescent/gradientdescent.h \
    gradientdescent/gradientdescentsimple/gradientdescentsimple.h \
    gradientdescent/gradientdescentadam/gradientdescentadam.h
