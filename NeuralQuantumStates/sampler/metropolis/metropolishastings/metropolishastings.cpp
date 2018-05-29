#include "metropolishastings.h"
#include "iostream"

using std::normal_distribution;
using std::shared_ptr;
using Eigen::VectorXd;

MetropolisHastings::MetropolisHastings(double dt, shared_ptr<NeuralQuantumState> nqs, int seed) :
    Metropolis(nqs, seed) {

    m_diffusionConstant = 0.5;
    m_dt                = dt;
    m_distributionXi    = normal_distribution<double>(0.0, 2*m_diffusionConstant);

    m_quantumForceCurrent.resize(m_ndim);
    m_sigmoidQTrial      .resize(m_nqs->getNH());
    m_positionCurrent    .resize(m_nqs->getNX());
}

/*
void MetropolisHastings::setTrialSample() {
    // Set i kontruktor
    std::uniform_int_distribution<> mrand(0, m_nqs->getNX()-1);
    m_updateCoordinate = mrand(m_randomEngine);

    m_xTrial                   = m_nqs->getX();
    m_quantumForceCurrent      = m_nqs->quantumForce(m_updateCoordinate);
    double xi                  = m_distributionXi(m_randomEngine);
    m_xCurrent                 = m_xTrial(m_updateCoordinate);
    m_xTrial(m_updateCoordinate) = m_xCurrent + m_diffusionConstant*m_quantumForceCurrent*m_dt + xi*sqrt(m_dt);

}*/

void MetropolisHastings::setTrialSample() {

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

/*
double MetropolisHastings::proposalRatio() {

    m_sigmoidQTrial           = m_nqs->computeSigmoidQ(m_QTrial);
    double quantumForceTrial  = m_nqs->quantumForce(m_updateCoordinate, m_xTrial, m_QTrial, m_sigmoidQTrial);
    //Greens function ratio
    double part1       = m_xCurrent - m_xTrial(m_updateCoordinate)
            - m_dt*m_diffusionConstant*quantumForceTrial;

    double part2       = m_xTrial(m_updateCoordinate) - m_xCurrent
            - m_dt*m_diffusionConstant*m_quantumForceCurrent;
    return exp(-(part1*part1 - part2*part2)/(4*m_diffusionConstant*m_dt));
}*/


double MetropolisHastings::proposalRatio() {

    m_sigmoidQTrial = m_nqs->computeSigmoidQ(m_QTrial);

    VectorXd quantumForceTrial(m_ndim);
    double   greensRatioExponent = 0;
    double   part1;
    double   part2;

    for (int d=0; d<m_ndim; d++) {
        quantumForceTrial(d) =
                m_nqs->quantumForce(m_ndim*m_particle+d, m_positionTrial, m_sigmoidQTrial);

        part1 = m_positionCurrent(m_ndim*m_particle+d) - m_positionTrial(m_ndim*m_particle+d)
                - m_dt*m_diffusionConstant*quantumForceTrial(d);
        part2 = m_positionTrial(m_ndim*m_particle+d) - m_positionCurrent(m_ndim*m_particle+d)
                - m_dt*m_diffusionConstant*m_quantumForceCurrent(d);

        greensRatioExponent += -part1*part1*+part2*part2;
    }

    return exp(greensRatioExponent/(4*m_diffusionConstant*m_dt));
}

/*
void MetropolisHastings::acceptTrialSample() {
    m_nqs->setX(m_xTrial);
    m_nqs->setPsi(m_psiTrial);
    m_nqs->setPsiComponents(m_QTrial, m_sigmoidQTrial);
}*/

void MetropolisHastings::acceptTrialSample() {
    m_nqs->setX(m_positionTrial);
    m_nqs->setInverseDistances(m_particle);
    m_nqs->setPsi(m_psiTrial);
    m_nqs->setPsiComponents(m_QTrial, m_sigmoidQTrial);
}
