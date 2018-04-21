#include "sampler.h"
#include <iostream>

Sampler::Sampler(int nSamples, int nCycles, Hamiltonian &hamiltonian,
                 NeuralQuantumState &nqs, Optimizer &optimizer) :
    m_hamiltonian(hamiltonian), m_nqs(nqs), m_optimizer(optimizer) {
    m_nSamples = nSamples;
    m_nCycles = nCycles;
    //m_hamiltonian = hamiltonian;
    //m_nqs = nqs;
    //m_optimizer = optimizer;

    std::random_device rd;
    m_randomEngine = std::mt19937_64(rd());
}

Sampler::Sampler(int nSamples, int nCycles, Hamiltonian &hamiltonian,
                 NeuralQuantumState &nqs, Optimizer &optimizer, int seed) :
    m_hamiltonian(hamiltonian), m_nqs(nqs), m_optimizer(optimizer) {
    m_nSamples = nSamples;
    m_nCycles = nCycles;
    //m_hamiltonian = hamiltonian;
    //m_nqs = nqs;
    //m_optimizer = optimizer;

    m_randomEngine = std::mt19937_64(seed);
}


void Sampler::runOptimizationSampling() {
    int nPar = m_nqs.m_nx + m_nqs.m_nh + m_nqs.m_nx*m_nqs.m_nh;
    double Eloc_temp;
    double variance;
    double acceptedRatio;
    // Wf derived wrt rbm parameters, to be added up for each sampling
    Eigen::VectorXd derPsi;
    Eigen::VectorXd derPsi_temp;
    derPsi.resize(nPar);
    derPsi_temp.resize(nPar);
    // Local energy times wf derived wrt rbm parameters, to be added up for each sampling
    Eigen::VectorXd EderPsi;
    EderPsi.resize(nPar);
    // The gradient
    Eigen::VectorXd grad;
    grad.resize(nPar);


    // Variables to store summations during sampling
    double Eloc;
    double Eloc2;
    double effectiveSamplings;
    int accepted;

    for (int cycles=0; cycles<m_nCycles; cycles++) {
        Eloc = 0;
        Eloc2 = 0;
        effectiveSamplings = 0;
        accepted = 0;
        derPsi.setZero();
        EderPsi.setZero();

        for (int samples=0; samples<m_nSamples; samples++) {
            samplePositions(accepted);
            if (samples > 0.1*m_nSamples) {
                Eloc_temp = m_hamiltonian.computeLocalEnergy(m_nqs);
                derPsi_temp = m_hamiltonian.computeLocalEnergyGradientComponent(m_nqs);

                // Add up values for expectation values
                Eloc += Eloc_temp;
                Eloc2 += Eloc_temp*Eloc_temp;
                effectiveSamplings++;
                derPsi += derPsi_temp;
                EderPsi += Eloc_temp*derPsi_temp;

            }
        }

        // Compute expectation values
        Eloc = Eloc/effectiveSamplings;
        Eloc2 = Eloc2/effectiveSamplings;
        derPsi = derPsi/effectiveSamplings;
        EderPsi = EderPsi/effectiveSamplings;


        variance = Eloc2 - Eloc*Eloc;
        acceptedRatio = accepted/(double)effectiveSamplings;

        // Compute gradient
        grad = 2*(EderPsi - Eloc*derPsi);

        // Update weights
        m_optimizer.optimizeWeights(m_nqs, grad, cycles);

        std::cout << cycles << "   " << Eloc << "   " << variance << "   " //<< a(0) << "   "
             << acceptedRatio << //" "
             //<< a(1) << " "
             //<< b(0) << " " << b(1) << " " << b(2) << " " << b(3) << " "
             //<< w(0,0) << " " << w(0,1) << " " << w(0,2) << " " << w(0,3) << " "
             //<< w(1,0) << " " << w(1,1) << " " << w(1,2) << " " << w(1,3) <<
               std::endl;
    }
}
