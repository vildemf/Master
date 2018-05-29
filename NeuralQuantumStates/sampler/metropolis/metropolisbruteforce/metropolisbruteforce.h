#ifndef METROPOLISBRUTEFORCE_H
#define METROPOLISBRUTEFORCE_H

#include "sampler/metropolis/metropolis.h"


class MetropolisBruteForce : public Metropolis {
private:
    std::uniform_real_distribution<double> m_distributionStep;
    double                                 m_step;

public:
    MetropolisBruteForce(double step, std::shared_ptr<NeuralQuantumState> nqs, int seed);

    void   setTrialSample();
    double proposalRatio();
    void   acceptTrialSample();
};

#endif // METROPOLISBRUTEFORCE_H
