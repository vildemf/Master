#include "neuralquantumstate.h"
#include <random>

NeuralQuantumState::NeuralQuantumState(int nh, int nx, int dim, double sigma) {
    m_nx = nx;
    m_nh = nh;
    m_dim  = dim;
    m_sig = sigma;
    m_sig2 = sigma*sigma;
    m_x.resize(m_nx); // positions/visibles
    m_h.resize(m_nh); // hidden
    m_a.resize(m_nx); // visible bias
    m_b.resize(m_nh); // hidden bias
    m_w.resize(m_nx, m_nh); // weights
    setupWeights();
    setupPositions();
}

void NeuralQuantumState::setupWeights() {
    int seed_initRBM=12345;
    float sigma_initRBM = 0.001;
    std::default_random_engine generator_initRBM(seed_initRBM);
    std::normal_distribution<double> distribution_initRBM(0,sigma_initRBM);
    for (int i=0; i<m_nx; i++){
        m_a(i) = distribution_initRBM(generator_initRBM);
        //outfile << a(i) << " ";
    }
    for (int i=0; i<m_nh; i++){
        m_b(i) = distribution_initRBM(generator_initRBM);
        //outfile << b(i) << " ";
    }
    for (int i=0; i<m_nx; i++){
        for (int j=0; j<m_nh; j++){
            m_w(i,j) = distribution_initRBM(generator_initRBM);
            //outfile << w(i,j) << " ";
        }
    }
    //outfile << '\n';
}

void NeuralQuantumState::setupPositions() {
    std::uniform_real_distribution<double> distribution_initX(-0.5,0.5);
    std::mt19937 rgen_Gibbs;
    std::random_device rd_Gibbs;
    rgen_Gibbs.seed(rd_Gibbs());
    for(int i=0; i<m_nx; i++){
        m_x(i)=distribution_initX(rgen_Gibbs);
    }
}

double NeuralQuantumState::computePsi() {
    m_psiFactor1 = 0.0;
    for (int i=0; i<m_nx; i++) {
        m_psiFactor1 += (m_x(i) - m_a(i))*(m_x(i) - m_a(i));
    }
    m_psiFactor2 = 1.0;
    m_Q = m_b + (((1.0/m_sig2)*m_x).transpose()*m_w).transpose();
    for (int j=0; j<m_nh; j++) {
        m_psiFactor2 *= (1 + exp(m_Q(j)));
    }
    m_psiFactor1 = exp(-m_psiFactor1/(2.0*m_sig2));
    return m_psiFactor1*m_psiFactor2;
}
