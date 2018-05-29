#include "hamiltonian.h"
#include "iostream"

using Eigen::VectorXd;

Hamiltonian::Hamiltonian(double omega, bool includeInteraction) {
    m_omega              = omega;
    m_includeInteraction = includeInteraction;
}


double Hamiltonian::computeLocalEnergy(NeuralQuantumState &nqs) {
    int nx                    = nqs.getNX();
    int ndim                  = nqs.getNDim();
    double kinetic            = nqs.computeLaplacian();
    double harmonicoscillator = harmonicOscillatorPotential(nqs.getX());
    double localEnergy        = 0.5*(kinetic + harmonicoscillator);

    // With Couloumb interaction:
    if (m_includeInteraction) {
        localEnergy          += nqs.getInverseDistances().sum();
    }

    return localEnergy;
}


double Hamiltonian::harmonicOscillatorPotential(VectorXd x) {
    return x.dot(m_omega*m_omega*x);
}

double Hamiltonian::getLocalEnergy() {
    return m_localEnergy;
}

void Hamiltonian::setLocalEnergy(double localEnergy) {
    m_localEnergy = localEnergy;
}
