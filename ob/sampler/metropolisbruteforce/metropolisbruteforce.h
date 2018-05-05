#ifndef METROPOLISBRUTEFORCE_H
#define METROPOLISBRUTEFORCE_H

#include "sampler/sampler.h"

class MetropolisBruteForce : public Sampler {
private:
    double m_step;
    double m_accepted;
    std::uniform_real_distribution<double> m_distributionStep;
    std::uniform_real_distribution<double> m_distributionTest;
    std::normal_distribution<double> m_distributionImportance;
public:
    MetropolisBruteForce(int nSamples, int nCycles, double step, Hamiltonian &hamiltonian,
               NeuralQuantumState &nqs, Optimizer &optimizer,
               std::string filename, std::string blockFilename, int seed);
    void samplePositions(bool &accepted);
};

#endif // METROPOLISBRUTEFORCE_H
