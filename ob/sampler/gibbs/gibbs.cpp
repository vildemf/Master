#include "gibbs.h"

Gibbs::Gibbs(int nSamples, int nCycles, Hamiltonian &hamiltonian,
             NeuralQuantumState &nqs, Optimizer &optimizer, std::string filename) :
    Sampler(nSamples, nCycles, hamiltonian, nqs, optimizer, filename) {
    m_distributionH = std::uniform_real_distribution<double>(0,1);
}

Gibbs::Gibbs(int nSamples, int nCycles, Hamiltonian &hamiltonian,
             NeuralQuantumState &nqs, Optimizer &optimizer, std::string filename, int seed) :
    Sampler(nSamples, nCycles, hamiltonian, nqs, optimizer, filename, seed) {
    m_distributionH = std::uniform_real_distribution<double>(0,1);
}


void Gibbs::samplePositions(int &accepted) {
    // set up
    Eigen::VectorXd probHgivenX;
    probHgivenX.resize(m_nqs.m_nh);

    for (int j=0; j<m_nqs.m_nh; j++) {
        probHgivenX(j) = 1.0/(1 + exp(-(m_nqs.m_b(j) + (((1.0/m_nqs.m_sig2)*m_nqs.m_x).transpose()*m_nqs.m_w.col(j)))));
        m_nqs.m_h(j) = m_distributionH(m_randomEngine) < probHgivenX(j);
        //if (cycles==59 && samples > 0.1*n_samples) {
            //outfile2 << h(j) << " ";
        //}
        //cout << h(j) << endl;
    }
    // Set new positions (visibles) given hidden, according to normal distribution
    std::normal_distribution<double> distributionX;
    double x_mean;
    for (int i=0; i<m_nqs.m_nx; i++) {
        x_mean = m_nqs.m_a(i) + m_nqs.m_w.row(i)*m_nqs.m_h;
        //cout << a(i) << "  " << x_mean << endl;
        distributionX = std::normal_distribution<double>(x_mean, m_nqs.m_sig);
        //cout << cycles << "   " << samples << "   " << i << "   " << x_mean << endl;
        m_nqs.m_x(i) = distributionX(m_randomEngine);
        //cout << cycles << "  " << samples << "  " << x(i) << "  " << x_mean << endl;
        //if (cycles==59 && samples > 0.1*n_samples) {
            //outfile2 << x(i) << " ";
        //}
    }
}
