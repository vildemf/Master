#include "montecarlomethod.h"

using Eigen::VectorXd;
using std::string;
using std::shared_ptr;
using std::fstream;

MonteCarloMethod::MonteCarloMethod(int numberOfSamples, shared_ptr<QuantumModel> model, int seed) :
    m_model(model) {
    /*
     * The constructor initializes a Monte Carlo method using Gibbs sampling.
     */

    m_sampleOneBodyDensities   = false;
    m_writeEnergiesForBlocking = false;
    m_numberOfSamples          = numberOfSamples;
    m_model                    ->setGibbsSampler(seed);

}

MonteCarloMethod::MonteCarloMethod(int numberOfSamples, shared_ptr<QuantumModel> model, int seed,
                                   string samplertype, double step) : m_model(model){
    /*
     * The constructor initializes a Monte Carlo method using Metropolis sampling.
     */

    m_sampleOneBodyDensities   = false;
    m_writeEnergiesForBlocking = false;
    m_numberOfSamples          = numberOfSamples;
    m_model                    ->setMetropolisSampler(seed, samplertype, step);

}


void MonteCarloMethod::runMonteCarlo() {
    /*
     * The function runs a Monte Carlo simulation for the given model m_model.
     */

    int nbins = 200;
    double rmin  = 0.0;
    double rmax  = 7.0;

    double binwidth = (rmax - rmin)/nbins;
    VectorXd onebodydensities(nbins);
    onebodydensities.setZero();


    m_model->setUpForSampling();
    int  effectiveNumberOfSamples = 0;
    bool equilibration;

    if (m_writeEnergiesForBlocking) {
        m_energiesblockingfile.open(m_energiesblockingfilename, std::fstream::out);
    }

    for (int sample=0; sample<m_numberOfSamples; sample++) {
        m_model->sample();
        equilibration = sample > 0.1*m_numberOfSamples;

        if (equilibration) {
            m_model->accumulateData();
            effectiveNumberOfSamples++;

            if (m_writeEnergiesForBlocking) {
                m_energiesblockingfile << m_model->getSampleEnergy() << "\n";
            }
            if (m_sampleOneBodyDensities) {
                m_model->sampleOneBodyDensities(rmax, rmin, binwidth, onebodydensities);
            }
        }

    }

    if (m_sampleOneBodyDensities) {
        fstream onebody;
        onebody.open(m_onebodydensitiesfilename, std::fstream::out);
        for (int i=0; i<nbins; i++) {
            onebody << onebodydensities(i) << "\n";
        }
        onebody.close();
    }

    if (m_writeEnergiesForBlocking) {
        m_energiesblockingfile.close();
        m_writeEnergiesForBlocking = false;
    }

    m_model->computeExpectationValues(effectiveNumberOfSamples);
    m_model->printExpectationValues();
}

void MonteCarloMethod::setWriteEnergiesForBlocking(string filename) {
    /*
     * Call this function in order to write the energies sampled during a Monte Carlo sampling to file.
     *
     * Note: Should there also be a way for the user to turn this off again?
     */

    m_writeEnergiesForBlocking = true;
    m_energiesblockingfilename = filename;
}

void MonteCarloMethod::setWriteOneBodyDensities(string filename) {
    /*
     * Call this function in order to sample one body densities and write to file.
     *
     * Note: Should there also be a way for the user to turn this off again?
     */

    m_sampleOneBodyDensities = true;
    m_onebodydensitiesfilename = filename;
}
