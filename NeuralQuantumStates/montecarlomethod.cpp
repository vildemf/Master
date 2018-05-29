#include "montecarlomethod.h"

using Eigen::VectorXd;
using std::string;
using std::shared_ptr;

MonteCarloMethod::MonteCarloMethod(int numberOfSamples, shared_ptr<QuantumModel> model, int seed) :
    m_model(model) {

    // constructor when using Gibbs sampling

    m_writeEnergiesForBlocking = false;
    m_numberOfSamples          = numberOfSamples;
    m_model                    ->setGibbsSampler(seed);

}

MonteCarloMethod::MonteCarloMethod(int numberOfSamples, shared_ptr<QuantumModel> model, int seed,
                                   string samplertype, double step) : m_model(model){
    // constructor when using Metropolis sampling
    // The idea is now that this should point to same object of model as a pointer outsiden this class (ie not
    // a copy). We point to model with a shared pointer for this reason.

    m_writeEnergiesForBlocking = false;
    m_numberOfSamples          = numberOfSamples;
    m_model                    ->setMetropolisSampler(seed, samplertype, step);

}


void MonteCarloMethod::runMonteCarlo() {

    m_model->setUpForSampling();
    int  effectiveNumberOfSamples = 0;
    bool equilibration;

    if (m_writeEnergiesForBlocking) {
        m_energiesblockingfile.open(m_energiesblockingfilename, std::fstream::out);
    }

    for (int sample=0; sample<m_numberOfSamples; sample++) {
        m_model->sample();
        equilibration = sample > 0.1*m_numberOfSamples;

        if (equilibration && !m_writeEnergiesForBlocking) {
            m_model->accumulateData();
            effectiveNumberOfSamples++;

        } else if (equilibration && m_writeEnergiesForBlocking) {
            m_model->accumulateData();
            effectiveNumberOfSamples++;
            //std::cout << "hello" << std::endl;
            m_energiesblockingfile << m_model->getSampleEnergy() << "\n";
        }
    }

    if (m_writeEnergiesForBlocking) {
        m_energiesblockingfile.close();
        m_writeEnergiesForBlocking = false;
    }

    m_model->computeExpectationValues(effectiveNumberOfSamples); // !! equilibration! acceptance count!
    m_model->printExpectationValues();
}

void MonteCarloMethod::setWriteEnergiesForBlocking(string filename) {
    // provide way of stopping as well (setting to false)? do automatically in the closing if test?
    //
    m_writeEnergiesForBlocking = true;
    m_energiesblockingfilename = filename;


    // Open file
}
