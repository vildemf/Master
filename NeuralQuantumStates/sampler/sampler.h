#ifndef SAMPLER_H
#define SAMPLER_H

#include "neuralquantumstate/neuralquantumstatepositivedefinite/neuralquantumstatepositivedefinite.h"
#include "hamiltonian.h"
#include <memory>

class Sampler {
protected:
    std::mt19937_64 m_randomEngine;

public:
    Sampler(int seed);
    virtual void sample(bool &accepted) = 0;

};

#endif // SAMPLER_H
