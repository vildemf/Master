#include "metropolisbruteforce.h"
#include "iostream"

using std::uniform_real_distribution;
using std::shared_ptr;

MetropolisBruteForce::MetropolisBruteForce(double step, shared_ptr<NeuralQuantumState> nqs, int seed) :
    Metropolis(nqs, seed) {

    m_step             = step;
    m_distributionStep = uniform_real_distribution<double>(-0.5, 0.5);
}

/*void MetropolisBruteForce::setTrialSample() {
    // ! i contstructor
    std::uniform_int_distribution<> mrand(0, m_nqs->getNX()-1);
    int updateCoordinate         = mrand(m_randomEngine);
    m_xTrial                     = m_nqs->getX();
    m_xTrial(updateCoordinate)  += m_distributionStep(m_randomEngine)*m_step;
}*/



void MetropolisBruteForce::setTrialSample() {
    m_particle      = m_distributionParticles(m_randomEngine);
    m_positionTrial = m_nqs->getX();

    for (int d=0; d<m_ndim; d++) {
        m_positionTrial(m_particle*m_ndim + d) += m_distributionStep(m_randomEngine)*m_step;
    }
}

double MetropolisBruteForce::proposalRatio() {
    return 1.0;
}

/*
void MetropolisBruteForce::acceptTrialSample() {
    m_nqs->setX(m_xTrial);
    m_nqs->setPsi(m_psiTrial);
    m_nqs->setPsiComponents(m_QTrial);
}*/

void MetropolisBruteForce::acceptTrialSample() {
    m_nqs->setX(m_positionTrial);
    m_nqs->setInverseDistances(m_particle);
    m_nqs->setPsi(m_psiTrial);
    m_nqs->setPsiComponents(m_QTrial);
}
