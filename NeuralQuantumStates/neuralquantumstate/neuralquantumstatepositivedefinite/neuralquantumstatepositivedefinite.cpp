#include "neuralquantumstatepositivedefinite.h"
#include "iostream"

using Eigen::VectorXd;

using std::normal_distribution;
using std::string;

NeuralQuantumStatePositiveDefinite::NeuralQuantumStatePositiveDefinite(double sigma, int nparticles, int nh, int ndim,
                                                                       string initialization, int seed) :
    NeuralQuantumState(sigma, nparticles, nh, ndim, initialization, seed) {
    m_positiveDefiniteFactor = 0.5;
    m_h                      .resize(m_nh);
}


VectorXd NeuralQuantumStatePositiveDefinite::probHGivenX() {
    // conditional probability that h=1
    return m_sigmoidQ;
}
normal_distribution<double> NeuralQuantumStatePositiveDefinite::probXiGivenH(int i) {
    // Returns probability distribution for x_i
    double xMean = m_a(i) + m_w.row(i)*m_h;  
    return normal_distribution<double>(xMean, m_sig);
}

double NeuralQuantumStatePositiveDefinite::computePsi(const VectorXd &x, const VectorXd &Q) {
    return sqrt(this->NeuralQuantumState::computePsi(x, Q));
}



double NeuralQuantumStatePositiveDefinite::getH(int j) {
    return m_h(j);
}
void NeuralQuantumStatePositiveDefinite::setH(int j, double hj) {
    m_h(j) = hj;
}
