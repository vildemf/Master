#include "metropolisimportancesampling.h"
#include "iostream"

MetropolisImportanceSampling::MetropolisImportanceSampling(int nSamples, int nCycles, double step, Hamiltonian &hamiltonian,
                                                           NeuralQuantumState &nqs, Optimizer &optimizer,
                                                           std::string filename, std::string blockFilename, int seed) :
    Sampler(nSamples, nCycles, hamiltonian, nqs, optimizer, filename, blockFilename, seed) {
    m_accepted = 0.0;
    m_step = step;
    m_distributionStep = std::uniform_real_distribution<double>(-0.5, 0.5);
    m_distributionTest = std::uniform_real_distribution<double>(0.0, 1.0);
    m_distributionImportance = std::normal_distribution<double>(0.0, 1.0);

}


// Update one coordinate at a time (one coordinate pr one sampling)
void MetropolisImportanceSampling::samplePositions(bool &accepted) {
    // Quantities that are kept for computational efficiency
    Eigen::VectorXd QTrial;
    Eigen::VectorXd sigmoidQTrial;
    sigmoidQTrial.resize(m_nqs.m_nh);


    // Suggestion of new position xTrial
    Eigen::VectorXd xTrial = m_nqs.m_x;

    std::uniform_int_distribution<> mrand(0, m_nqs.m_nx-1);
    int updateCoordinate = mrand(m_randomEngine);

    double D = 0.5;
    double xi = m_distributionImportance(m_randomEngine);
    double Fcurrent;
    double Ftrial;

    // Compute quantum force
    Fcurrent = m_nqs.quantumForce(updateCoordinate);

    // Update coordinate
    double xCurrent = m_nqs.m_x(updateCoordinate);
    xTrial(updateCoordinate) = xCurrent + D*Fcurrent*m_step + xi*sqrt(m_step);
    QTrial = m_nqs.m_b + (1.0/m_nqs.m_sig2)*(xTrial.transpose()*m_nqs.m_w).transpose();

    Ftrial = m_nqs.quantumForce(updateCoordinate, xTrial, QTrial, sigmoidQTrial);

    //Greens ratio
    double part1 = xCurrent - xTrial(updateCoordinate) - m_step*Ftrial;
    double part2 = xTrial(updateCoordinate) - xCurrent - m_step*Fcurrent;
    double Gratio = exp(-(part1*part1 - part2*part2)/(4*D*m_step));

    // Psi
    double psiTrial = m_nqs.computePsi(xTrial, QTrial);
    double probCurrent = m_nqs.m_psi*m_nqs.m_psi;

    double probRatio = Gratio*psiTrial*psiTrial/probCurrent;

    //std::cout << accepted << "  " << probRatio << "  " << m_nqs.m_psi << "  " << psiTrial << "  " << Fcurrent
      //        << "  " << Ftrial << "  " << xCurrent
        //      << "  " << xTrial(updateCoordinate) << std::endl;

    if ((1.0 < probRatio) || (m_distributionTest(m_randomEngine) < probRatio)) {
        m_nqs.m_x = xTrial;
        m_nqs.m_psi = psiTrial;
        m_nqs.updatePsiComponents(QTrial, sigmoidQTrial);
        accepted=true;
    } else {
        accepted=false;
    }


}
