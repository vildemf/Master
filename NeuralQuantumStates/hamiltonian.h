#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H


#include "neuralquantumstate/nerualquantumstate.h"


class Hamiltonian {
private:
    double m_omega;
    double m_localEnergy;
    bool   m_includeInteraction;

public:
    Hamiltonian(double omega, bool includeInteraction);
    double computeLocalEnergy(const NeuralQuantumState &nqs);
    double harmonicOscillatorPotential(const Eigen::VectorXd &x);
    double getLocalEnergy();
    void   setLocalEnergy(double localEnergy);
};

#endif // HAMILTONIAN_H
