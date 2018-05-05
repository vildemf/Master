#ifndef METROPOLISIMPORTANCESAMPLING_H
#define METROPOLISIMPORTANCESAMPLING_H

#include "sampler/sampler.h"

class MetropolisImportanceSampling : public Sampler {
private:
    double m_step;
    double m_accepted;
    std::uniform_real_distribution<double> m_distributionStep;
    std::uniform_real_distribution<double> m_distributionTest;
    std::normal_distribution<double> m_distributionImportance;
public:
    MetropolisImportanceSampling(int nSamples, int nCycles, double step, Hamiltonian &hamiltonian,
        NeuralQuantumState &nqs, Optimizer &optimizer,
        std::string filename, std::string blockFilename, int seed);
    void samplePositions(bool &accepted);
};

#endif // METROPOLISIMPORTANCESAMPLING_H
