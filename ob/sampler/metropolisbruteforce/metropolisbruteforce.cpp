#include "metropolisbruteforce.h"
#include "math.h"


MetropolisBruteForce::MetropolisBruteForce(int nSamples, int nCycles, double step, Hamiltonian &hamiltonian,
                       NeuralQuantumState &nqs, Optimizer &optimizer,
                       std::string filename, std::string blockFilename, int seed) :
    Sampler(nSamples, nCycles, hamiltonian, nqs, optimizer, filename, blockFilename, seed) {
    m_accepted = 0.0;
    m_step = step;
    m_distributionStep = std::uniform_real_distribution<double>(-0.5, 0.5);
    m_distributionTest = std::uniform_real_distribution<double>(0.0, 1.0);
    m_distributionImportance = std::normal_distribution<double>(0.0, 1.0);
}

// Update all coordinates of all particles pr sampling
/*
void MetropolisBruteForce::samplePositions(int &accepted) {
    // Suggestion of new position xTrial
    Eigen::VectorXd xTrial;
    xTrial.resize(m_nqs.m_nx);
    for (int i=0; i<m_nqs.m_nx; i++) {
        xTrial(i) = m_nqs.m_x(i) + m_distributionStep(m_randomEngine)*m_step;
    }

    double psiTrial = m_nqs.computePsi(xTrial);

    double probCurrent = m_psi*m_psi;
    double probTrial = psiTrial*psiTrial;\
    double probRatio = probTrial/probCurrent;

    if ((1.0 < probRatio) || (m_distributionTest(m_randomEngine) < probRatio)) {
        m_nqs.m_x = xTrial;
        m_psi = psiTrial;
        //m_accepted++;
        accepted++;
    }
}
*/

// Update one coordinate at a time (one coordinate pr one sampling)
void MetropolisBruteForce::samplePositions(bool &accepted) {
    // Suggestion of new position xTrial
    Eigen::VectorXd xTrial = m_nqs.m_x;
    Eigen::VectorXd QTrial;


    std::uniform_int_distribution<> mrand(0, m_nqs.m_nx-1);
    int updateCoordinate = mrand(m_randomEngine);

    xTrial(updateCoordinate) += m_distributionStep(m_randomEngine)*m_step;
    QTrial = m_nqs.m_b + (1.0/m_nqs.m_sig2)*(xTrial.transpose()*m_nqs.m_w).transpose();

    double psiTrial = m_nqs.computePsi(xTrial, QTrial);

    double probCurrent = m_nqs.m_psi*m_nqs.m_psi;
    double probTrial = psiTrial*psiTrial;
    double probRatio = probTrial/probCurrent;

    if ((1.0 < probRatio) || (m_distributionTest(m_randomEngine) < probRatio)) {
        m_nqs.m_x = xTrial;
        m_nqs.m_psi = psiTrial;
        m_nqs.updatePsiComponents(QTrial);
        accepted=true;
    } else {
        accepted=false;
    }

}

