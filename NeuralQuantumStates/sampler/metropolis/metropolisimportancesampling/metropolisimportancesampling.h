#ifndef METROPOLISIMPORTANCESAMPLING_H
#define METROPOLISIMPORTANCESAMPLING_H

#include "sampler/metropolis/metropolis.h"

class MetropolisImportanceSampling : public Metropolis {
    std::normal_distribution<double> m_distributionXi;
    Eigen::VectorXd                  m_sigmoidQTrial;
    double                           m_diffusionConstant;
    double                           m_dt;

    Eigen::VectorXd                  m_quantumForceCurrent;
    Eigen::VectorXd                  m_positionCurrent;

    //int      m_updateCoordinate;
    //double   m_xCurrent;
    //double   m_quantumForceCurrent;



public:
    MetropolisImportanceSampling(double dt, std::shared_ptr<NeuralQuantumState> nqs, int seed);
    void   setTrialSample();
    double proposalRatio();
    void   acceptTrialSample();
};

#endif // METROPOLISImportanceSampling_H
