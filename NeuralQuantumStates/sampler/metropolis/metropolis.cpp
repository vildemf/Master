#include "metropolis.h"
#include "iostream"

using Eigen::VectorXd;
using std::shared_ptr;

Metropolis::Metropolis(shared_ptr<NeuralQuantumState> nqs, int seed) :
    Sampler(seed), m_nqs(nqs)  {

    m_distributionAcceptanceTest = std::uniform_real_distribution<double>(0.0, 1.0);
    m_distributionParticles      = std::uniform_int_distribution<>(0, m_nqs->getNParticles()-1);
    m_ndim                       = m_nqs->getNDim();

    m_positionTrial.resize(m_nqs->getNX());
    m_QTrial       .resize(m_nqs->getNH());
}


void Metropolis::sample(bool &accepted) {
    /*
     * The function samples a new position configuration accordin to the Metropolis method. Depending
     * on the derived class the method will either be brute force or importance sampled.
     */

    setTrialSample();

    // Obs: prob and prop has to be called in this sequence (for importance sampling)
    double prob       = probabilityRatio();
    double prop       = proposalRatio();
    double acceptance = prob*prop;
    accepted          = false;

    if (acceptance > m_distributionAcceptanceTest(m_randomEngine)) {
        accepted      = true;
        acceptTrialSample();
    }
}

double Metropolis::probabilityRatio() {
    /*
     * The function calculates the ratio between the probability of the trial state and current state.
     */

    m_nqs->computeQ(m_positionTrial, m_QTrial);

    m_psiTrial        = m_nqs->computePsi(m_positionTrial, m_QTrial);
    double psiCurrent = m_nqs->getPsi();

    return m_psiTrial*m_psiTrial/(psiCurrent*psiCurrent);
}

