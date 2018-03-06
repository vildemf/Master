#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "neuralquantumstate.h"

class Optimizer {
public:
    Optimizer();
    virtual void optimizeWeights() {}
};

#endif // OPTIMIZER_H
