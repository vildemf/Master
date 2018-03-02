#ifndef NEURALQUANTUMSTATE_H
#define NEURALQUANTUMSTATE_H

#include <Eigen/Dense>

class NeuralQuantumState {
private:
    int m_nx;
    int m_nh;
    double m_sig;
    double m_sig2;
    Eigen::VectorXd m_x;
    Eigen::VectorXd m_h;
    Eigen::VectorXd m_a;
    Eigen::VectorXd m_b;
    Eigen::MatrixXd m_w;

    double m_psiFactor1;
    double m_psiFactor2;
    Eigen::VectorXd m_Q;

    void setupWeights();
    void setupPositions();
    double computePsi();

public:
    NeuralQuantumState(int nh, int nx, double sigma);
};

#endif // NEURALQUANTUMSTATE_H
