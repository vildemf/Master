#ifndef QUANTUMMODEL_H
#define QUANTUMMODEL_H


#include "sampler/gibbs/gibbs.h"
#include "sampler/metropolis/metropolisbruteforce/metropolisbruteforce.h"
#include "sampler/metropolis/metropolishastings/metropolishastings.h"

class QuantumModel {
private:
    //Hamiltonian *m_hamiltonian;
    //NeuralQuantumState *m_nqs;
    //Sampler *m_sampler;
    std::shared_ptr<NeuralQuantumState> m_nqs;
    std::unique_ptr<Hamiltonian>        m_hamiltonian;
    std::unique_ptr<Sampler>            m_sampler;
    int                                 m_nx;
    int                                 m_nh;
    int                                 m_ndim;
    int                                 m_nparticles;
    int                                 m_nparameters;
    std::string                         m_wavefunctiontype;
    bool                                m_accepted;

    // Accumulated/Expected values
    double                              m_localEnergy;
    double                              m_localEnergySquared;
    double                              m_variance;
    double                              m_acceptcount;
    double                              m_localEnergyGradientNorm;
    Eigen::VectorXd                     m_dPsi; // 1/psi * dPsi/dalpha_i
    Eigen::VectorXd                     m_localEnergydPsi;
    Eigen::VectorXd                     m_localEnergyGradient;

    void initializeWavefunction(std::string nqsType, std::string nqsInitialization, int nqsSeed);

public:
    QuantumModel(double harmonicoscillatorOmega, bool couloumbinteraction, std::string nqsType, std::string nqsInitialization,
                 int nParticles, int nDimensions, int nHidden, int nqsSeed);

    void            setGibbsSampler(int seed);
    void            setMetropolisSampler(int seed, std::string samplertype, double step);

    void            setAccumulativeDataToZero();
    void            setUpForSampling();
    void            sample();
    void            accumulateData();
    void            computeExpectationValues(int numberOfSamples);
    void            printExpectationValues();

    void            shiftParameters(Eigen::VectorXd shift);

    void            writeParametersToFile(std::string filename);

    double          getSampleEnergy();
    double          getExpectedLocalEnergy();
    double          getVariance();
    double          getGradientNorm();
    int             getNParameters();
    Eigen::VectorXd getGradient();

};

#endif // QUANTUMMODEL_H
