TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    neuralquantumstate.cpp \
    hamiltonian.cpp \
    sampler/sampler.cpp \
    optimizer/optimizer.cpp \
    sampler/gibbs/gibbs.cpp \
    sampler/metropolisbruteforce/metropolisbruteforce.cpp \
    sampler/metropolisimportancesampling/metropolisimportancesampling.cpp \
    optimizer/sgd/sgd.cpp \
    optimizer/asgd/asgd.cpp \
    sampler/metropolis.cpp

HEADERS += \
    neuralquantumstate.h \
    hamiltonian.h \
    sampler/sampler.h \
    optimizer/optimizer.h \
    sampler/gibbs/gibbs.h \
    sampler/metropolisbruteforce/metropolisbruteforce.h \
    sampler/metropolisimportancesampling/metropolisimportancesampling.h \
    optimizer/sgd/sgd.h \
    optimizer/asgd/asgd.h \
    sampler/metropolis.h

INCLUDEPATH += /usr/local/include/eigen3/
