#ifndef NEURALQUANTUMSTATE_H
#define NEURALQUANTUMSTATE_H

#include <Eigen/Dense>
#include <random>

class NeuralQuantumState {
private:
    std::mt19937_64 m_randomEngine; // For the distributions

    void setup(int nh, int nx, int dim, double sigma, bool gaussianInitialization);
    void setupWeights();
    void setupPositions();

public:
    int m_nx;
    int m_nh;
    int m_dim;
    double m_sig;
    double m_sig2;
    Eigen::VectorXd m_x;
    Eigen::VectorXd m_h;
    Eigen::VectorXd m_a;
    Eigen::VectorXd m_b;
    Eigen::MatrixXd m_w;

    // Quantities to store for computational efficiency
    double m_psi;
    Eigen::VectorXd m_sigmoidQ;
    Eigen::VectorXd m_derSigmoidQ;

    NeuralQuantumState(int nh, int nx, int dim, double sigma, bool gaussianInitialization, int seed);

    double computePsi(Eigen::VectorXd x, Eigen::VectorXd Q); // Needed for Sampler Metropolis method

    double quantumForce(int updateCoordinate);
    double quantumForce(int updateCoordinate, Eigen::VectorXd xTrial, Eigen::VectorXd Q, Eigen::VectorXd &sigmoidQ);

    void updatePsiComponents();
    void updatePsiComponents(Eigen::VectorXd Q);
    void updatePsiComponents(Eigen::VectorXd Q, Eigen::VectorXd sigmoidQ);
};

#endif // NEURALQUANTUMSTATE_H
