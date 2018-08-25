#include "metropolisimportancesampling.h"
#include "iostream"

using std::normal_distribution;
using std::shared_ptr;
using Eigen::VectorXd;

MetropolisImportanceSampling::MetropolisImportanceSampling(double dt, shared_ptr<NeuralQuantumState> nqs, int seed) :
    Metropolis(nqs, seed) {

    m_diffusionConstant = 0.5;
    m_dt                = dt;
    m_distributionXi    = normal_distribution<double>(0.0, 2*m_diffusionConstant);

    m_quantumForceCurrent.resize(m_ndim);
    m_sigmoidQTrial      .resize(m_nqs->getNH());
    m_positionCurrent    .resize(m_nqs->getNX());
}


void MetropolisImportanceSampling::setTrialSample() {
    /*
     * The function sets a trial position configuration according to the importance sampling method.
     */

    m_particle        = m_distributionParticles(m_randomEngine);
    m_positionTrial   = m_nqs->getX();
    m_positionCurrent = m_nqs->getX();

    double xi;
    for (int d=0; d<m_ndim; d++) {
        m_quantumForceCurrent(d) = m_nqs->quantumForce(m_ndim*m_particle + d);
        xi                       = m_distributionXi(m_randomEngine);

        m_positionTrial(m_ndim*m_particle + d)
                = m_positionCurrent(m_ndim*m_particle + d)
                + m_diffusionConstant*m_quantumForceCurrent(d)*m_dt
                + xi*sqrt(m_dt);
    }
}


double MetropolisImportanceSampling::proposalRatio() {
    /*
     * The function computes the proposal ratio according to the importance sampling method, using Green
     * functions.
     */

    m_nqs->computeSigmoidQ(m_QTrial, m_sigmoidQTrial);

    VectorXd quantumForceTrial(m_ndim);
    double   greensRatioExponent = 0;
    double   part1;
    double   part2;

    double gf = 0.0;
    for (int p=0; p<m_nqs->getNX(); p+=m_ndim) {
        for (int d=0; d<m_ndim; d++) {
            quantumForceTrial(d) =
                    m_nqs->quantumForce(p+d, m_positionTrial, m_sigmoidQTrial);
            m_quantumForceCurrent(d) =
                    m_nqs->quantumForce(p+d);
            gf += 0.5 *
                    ( m_quantumForceCurrent(d) + quantumForceTrial(d) ) *
                    (m_diffusionConstant * m_dt * 0.5 *
                     (m_quantumForceCurrent(d) - quantumForceTrial(d)) -
                     m_positionTrial(p+d) + m_positionCurrent(p+d)  );

        }
    }
    return exp(gf);
}


void MetropolisImportanceSampling::acceptTrialSample() {
    /*
     * The function performs the updates necessary when a trial state is accepted.
     */

    m_nqs->setX(m_positionTrial);
    m_nqs->setInverseDistances(m_particle);
    m_nqs->setPsi(m_psiTrial);
    m_nqs->setPsiComponents(m_QTrial, m_sigmoidQTrial);
}
