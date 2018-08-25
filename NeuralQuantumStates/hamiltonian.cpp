#include "hamiltonian.h"
#include "iostream"

using Eigen::VectorXd;

Hamiltonian::Hamiltonian(double omega, bool includeInteraction) {
    /*
     * Constructor of the Hamiltonian class. The current implementation
     * handles Hamiltonians with harmonic oscilator potentials and with
     * and without Coulomb interaction. The class can be generalized to
     * handle more Hamiltonians.
     */
    m_omega              = omega;
    m_includeInteraction = includeInteraction;
}


double Hamiltonian::computeLocalEnergy(const NeuralQuantumState &nqs) {
    /*
     * The function computes the energy of the system state
     * described by the given NQS wavefunction object.
     */
    int nx                    = nqs.getNX();
    int ndim                  = nqs.getNDim();
    double kinetic            = nqs.computeLaplacian();
    double harmonicoscillator = harmonicOscillatorPotential(nqs.getX());
    double localEnergy        = 0.5*(kinetic + harmonicoscillator);

    // With Coulomb interaction:
    if (m_includeInteraction) {
        localEnergy          += nqs.getInverseDistances().sum();
    }

    return localEnergy;
}


double Hamiltonian::harmonicOscillatorPotential(const VectorXd &x) {
    /*
     * The function computes the harmonic oscillator potential energy of
     * the system for a given configuration of particle positions.
     */
    return x.dot(m_omega*m_omega*x);
}

double Hamiltonian::getLocalEnergy() {
    /*
     * The function returns the currently stored local energy.
     */
    return m_localEnergy;
}

void Hamiltonian::setLocalEnergy(double localEnergy) {
    /*
     * The function updates the currently stored local energy.
     */
    m_localEnergy = localEnergy;
}
