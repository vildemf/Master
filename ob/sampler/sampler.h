#ifndef SAMPLER_H
#define SAMPLER_H


class Sampler {
private:
    int m_nSamples;
    int m_nCycles;
    int m_nDim;
    int m_nx;
    int m_nh;

    class Hamiltonian* m_hamiltonian;
    class NerualQuantumState* m_nqs;
    class Minimizer* m_minimizer;

public:
    Sampler();
};

#endif // SAMPLER_H
