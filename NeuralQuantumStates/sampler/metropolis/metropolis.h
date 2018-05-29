#ifndef METROPOLIS_H
#define METROPOLIS_H

#include "sampler/sampler.h"

class Metropolis : public Sampler {
private:
    std::uniform_real_distribution<double> m_distributionAcceptanceTest;

protected:
    std::uniform_int_distribution<>        m_distributionParticles;
    std::shared_ptr<NeuralQuantumState>    m_nqs;
    //Eigen::VectorXd                        m_xTrial;
    Eigen::VectorXd                        m_QTrial;
    double                                 m_psiTrial;

    Eigen::VectorXd                        m_positionTrial;
    int                                    m_particle;
    int                                    m_ndim;

public:
    Metropolis(std::shared_ptr<NeuralQuantumState> nqs, int seed);
    void           sample(bool &accepted);

    double         probabilityRatio();
    virtual void   setTrialSample() = 0;
    virtual double proposalRatio() = 0;
    virtual void   acceptTrialSample() = 0;
};

#endif // METROPOLIS_H
