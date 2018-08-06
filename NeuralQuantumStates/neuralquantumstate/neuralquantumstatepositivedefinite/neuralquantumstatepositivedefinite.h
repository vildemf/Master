#ifndef NEURALQUANTUMSTATEPOSITIVEDEFINITE_H
#define NEURALQUANTUMSTATEPOSITIVEDEFINITE_H

#include "neuralquantumstate/nerualquantumstate.h"

class NeuralQuantumStatePositiveDefinite  : public NeuralQuantumState {
private:
    Eigen::VectorXd m_h;

public:
    NeuralQuantumStatePositiveDefinite(double sigma, int nparticles, int nh, int ndim, std::string initialization, int seed);

    double                           computePsi(const Eigen::VectorXd &x, const Eigen::VectorXd &Q);

    Eigen::VectorXd                  probHGivenX();
    std::normal_distribution<double> probXiGivenH(int i);

    double                           getH(int j);
    void                             setH(int j, double hj);
};

#endif // NEURALQUANTUMSTATEPOSITIVEDEFINITE_H
