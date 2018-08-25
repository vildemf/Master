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
    /*
     * The function returns the conditional probabilities that h_j=1, given the values of m_x.
     */

    return m_sigmoidQ;
}
normal_distribution<double> NeuralQuantumStatePositiveDefinite::probXiGivenH(int i) {
    /*
     * The function returns the conditional probability distribution of each x_i, given the
     * values of m_h.
     */

    double xMean = m_a(i) + m_w.row(i)*m_h;  
    return normal_distribution<double>(xMean, m_sig);
}

double NeuralQuantumStatePositiveDefinite::computePsi(const VectorXd &x, const VectorXd &Q) {
    /*
     * The function returns the value of the wavefunction.
     */

    return sqrt(this->NeuralQuantumState::computePsi(x, Q));
}



double NeuralQuantumStatePositiveDefinite::getH(int j) {
    return m_h(j);
}
void NeuralQuantumStatePositiveDefinite::setH(int j, double hj) {
    m_h(j) = hj;
}
