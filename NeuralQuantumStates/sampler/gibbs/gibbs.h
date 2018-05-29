#ifndef GIBBS_H
#define GIBBS_H

#include "sampler/sampler.h"

class Gibbs : public Sampler {
private:
    std::uniform_real_distribution<double>              m_distributionH;
    std::shared_ptr<NeuralQuantumStatePositiveDefinite> m_nqs;

public:
    Gibbs(std::shared_ptr<NeuralQuantumStatePositiveDefinite> nqs, int seed);
    void sample(bool &accepted);
};


#endif // GIBBS_H
