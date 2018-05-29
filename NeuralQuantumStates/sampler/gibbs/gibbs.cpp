#include "gibbs.h"

using Eigen::VectorXd;
using std::normal_distribution;
using std::shared_ptr;

Gibbs::Gibbs(shared_ptr<NeuralQuantumStatePositiveDefinite> nqs, int seed) :
    Sampler(seed), m_nqs(nqs) {

    m_distributionH = std::uniform_real_distribution<double>(0,1);
}


void Gibbs::sample(bool &accepted) {
    // Set new hidden variables given positions, according to the logistic sigmoid function
    // (implemented by comparing the sigmoid probability to a uniform random variable)
    VectorXd probHGivenX = m_nqs->probHGivenX();

    for (int j=0; j<m_nqs->getNH(); j++) {
        m_nqs->setH(j, (probHGivenX(j) > m_distributionH(m_randomEngine)));
    }

    // Set new positions (visibles) given hidden, according to normal distribution
    normal_distribution<double> distributionX;

    for (int i=0; i<m_nqs->getNX(); i++) {
        distributionX = m_nqs->probXiGivenH(i);
        m_nqs->setX(i, distributionX(m_randomEngine));
    }

    m_nqs->setPsiComponents();
    m_nqs->setInverseDistances();
    accepted=true;
}
