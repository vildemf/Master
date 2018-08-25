#include "metropolisbruteforce.h"
#include "iostream"

using std::uniform_real_distribution;
using std::shared_ptr;

MetropolisBruteForce::MetropolisBruteForce(double step, shared_ptr<NeuralQuantumState> nqs, int seed) :
    Metropolis(nqs, seed) {

    m_step             = step;
    m_distributionStep = uniform_real_distribution<double>(-0.5, 0.5);
}


void MetropolisBruteForce::setTrialSample() {
    /*
     * The function sets a trial position configuration according to the brute force method.
     */

    m_particle      = m_distributionParticles(m_randomEngine);
    m_positionTrial = m_nqs->getX();

    for (int d=0; d<m_ndim; d++) {
        m_positionTrial(m_particle*m_ndim + d) += m_distributionStep(m_randomEngine)*m_step;
    }
}

double MetropolisBruteForce::proposalRatio() {
    /*
     * The proposal ratio of the brute force method is simply 1.
     */

    return 1.0;
}

void MetropolisBruteForce::acceptTrialSample() {
    /*
     * The function performs the updates necessary when a trial state is accepted.
     */

    m_nqs->setX(m_positionTrial);
    m_nqs->setInverseDistances(m_particle);
    m_nqs->setPsi(m_psiTrial);
    m_nqs->setPsiComponents(m_QTrial);
}
