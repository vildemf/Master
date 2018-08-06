#include "quantummodel.h"
#include "iostream"
#include <fstream>

using Eigen::VectorXd;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;
using std::static_pointer_cast;
using std::move;

QuantumModel::QuantumModel(double harmonicoscillatorOmega, bool coulombinteraction, string nqsType, string nqsInitialization,
                           int nParticles, int nDimensions, int nHidden, int nqsSeed) :
    m_hamiltonian(new Hamiltonian(harmonicoscillatorOmega, coulombinteraction)) {

    m_nx               = nParticles*nDimensions;
    m_nh               = nHidden;
    m_ndim             = nDimensions;
    m_nparticles       = nParticles;
    m_nparameters      = m_nx + m_nh + m_nx*m_nh;
    m_wavefunctiontype = nqsType;

    m_dPsi               .resize(m_nparameters);                   // 1/psi * dPsi/dalpha_i
    m_localEnergydPsi    .resize(m_nparameters);
    m_localEnergyGradient.resize(m_nparameters);

    initializeWavefunction(harmonicoscillatorOmega, nqsType, nqsInitialization, nqsSeed);
}


void QuantumModel::setUpForSampling() {
    setAccumulativeDataToZero();

    VectorXd Q(m_nh);

    m_nqs        ->computeQ(m_nqs->getX(), Q);
    m_nqs        ->setPsi(m_nqs->computePsi(m_nqs->getX(), Q));
    m_nqs        ->setPsiComponents();
    m_hamiltonian->setLocalEnergy(m_hamiltonian->computeLocalEnergy(*m_nqs));
    m_nqs        ->setParameterDerivative(m_nqs->computeParameterDerivative());
}

void QuantumModel::setAccumulativeDataToZero() {
    m_localEnergy        = 0;
    m_localEnergySquared = 0;
    m_acceptcount        = 0;

    m_dPsi           .setZero();
    m_localEnergydPsi.setZero();
}

void QuantumModel::sample() {
    m_sampler->sample(m_accepted);
}

void QuantumModel::accumulateData() {
    if (m_accepted) {
        m_nqs->setParameterDerivative(m_nqs->computeParameterDerivative());
        m_hamiltonian->setLocalEnergy(m_hamiltonian->computeLocalEnergy(*m_nqs));
        m_acceptcount++;
    }

    double localEnergy   = m_hamiltonian->getLocalEnergy();
    VectorXd dPsi        = m_nqs->getParamterDerivative();

    m_localEnergy        += localEnergy;
    m_localEnergySquared += localEnergy*localEnergy;
    m_dPsi               += dPsi;
    m_localEnergydPsi    += localEnergy*dPsi;
}


void QuantumModel::sampleOneBodyDensities(double rmax, double rmin, double binwidth, VectorXd &oneBodyDensities) {
    VectorXd x = m_nqs->getX();
    double r;
    int binindex;

    for (int p=0; p<m_nx; p+=m_ndim) {
        r = 0;
        for (int d=0; d<m_ndim; d++) {
            r += x(p + d)*x(p + d);
        }
        r = sqrt(r);
        if (rmin <= r && r < rmax) {
            binindex = floor((r - rmin)/binwidth);
            oneBodyDensities(binindex) += 1;
        }
    }
}


void QuantumModel::computeExpectationValues(int numberOfSamples) {
    m_localEnergy             = m_localEnergy          /numberOfSamples;
    m_localEnergySquared      = m_localEnergySquared   /numberOfSamples;
    m_dPsi                    = m_dPsi                 /numberOfSamples;
    m_localEnergydPsi         = m_localEnergydPsi      /numberOfSamples;
    m_acceptcount             = m_acceptcount          /numberOfSamples;

    m_variance                = (m_localEnergySquared - m_localEnergy*m_localEnergy) /numberOfSamples;

    m_localEnergyGradient     = 2*(m_localEnergydPsi - m_localEnergy*m_dPsi);
    m_localEnergyGradientNorm = sqrt(m_localEnergyGradient.dot(m_localEnergyGradient));
}


void QuantumModel::printExpectationValues() {
    std::cout << "------------------------------------------" << std::endl
              << "Local energy:          " << m_localEnergy << std::endl
              << "Standard error:        " << sqrt(m_variance)    << std::endl
              << "Acceptance ratio:      " << m_acceptcount << std::endl;
}


void QuantumModel::shiftParameters(const VectorXd &shift) {
    for (int i=0; i<m_nx; i++) {
        m_nqs->setA(i, m_nqs->getA(i)+shift(i));
    }
    for (int j=0; j<m_nh; j++) {
        m_nqs->setB(j, m_nqs->getB(j)+shift(m_nx + j));
    }
    int k = m_nx + m_nh;
    for (int i=0; i<m_nx; i++) {
        for (int j=0; j<m_nh; j++) {
            m_nqs->setW(i, j, m_nqs->getW(i,j)+shift(k));
            k++;
        }
    }
}



void QuantumModel::initializeWavefunction(double omega, string nqsType, string nqsInitialization, int nqsSeed) {
    if (nqsType=="general") {
        double sigma = 1./sqrt(omega);
        m_nqs = make_shared<NeuralQuantumState>(sigma, m_nparticles, m_nh, m_ndim, nqsInitialization, nqsSeed);
    } else if (nqsType=="positivedefinite") {
        double sigma = 1./sqrt(2*omega);
        m_nqs = make_shared<NeuralQuantumStatePositiveDefinite>(sigma, m_nparticles, m_nh, m_ndim,
                                                                nqsInitialization, nqsSeed);
    }

}




void QuantumModel::setGibbsSampler(int seed) {
    if (m_wavefunctiontype=="positivedefinite") {
        shared_ptr<NeuralQuantumStatePositiveDefinite> nqsPosDef
                = static_pointer_cast<NeuralQuantumStatePositiveDefinite>(m_nqs);
        unique_ptr<Sampler> sampler(new Gibbs(nqsPosDef, seed));
        m_sampler = move(sampler);
    } else {
        std::cout << "Error! The Gibbs sampler cannot be used for the general wavefunction type." << std::endl;
    }
}

void QuantumModel::setMetropolisSampler(int seed, string samplertype, double step) {
    if (samplertype=="bruteforce") {
        unique_ptr<Sampler> sampler(new MetropolisBruteForce(step, m_nqs, seed));
        m_sampler = move(sampler);
    } else if (samplertype=="importancesampling") {
        unique_ptr<Sampler> sampler(new MetropolisImportanceSampling(step, m_nqs, seed));
        m_sampler = move(sampler);
    }
}





double QuantumModel::getSampleEnergy() {
    return m_hamiltonian->getLocalEnergy();
}
int QuantumModel::getNParameters() {
    return m_nparameters;
}
double QuantumModel::getExpectedLocalEnergy() {
    return m_localEnergy;
}
double QuantumModel::getVariance() {
    return m_variance;
}

VectorXd QuantumModel::getGradient() {
    return m_localEnergyGradient;
}
double QuantumModel::getGradientNorm() {
    return m_localEnergyGradientNorm;
}






void QuantumModel::writeParametersToFile(string filename) {
    std::ofstream parameterfile;
    parameterfile.open(filename, std::ofstream::out);
    for (int i=0; i<m_nx; i++) {
        parameterfile << m_nqs->getA(i) << " ";
    }
    parameterfile << "\n";
    for (int j=0; j<m_nh; j++) {
        parameterfile << m_nqs->getB(j) << " ";
    }
    parameterfile << "\n";
    int k = m_nx + m_nh;
    for (int i=0; i<m_nx; i++) {
        for (int j=0; j<m_nh; j++) {
            parameterfile << m_nqs->getW(i, j) << " ";
            k++;
        }
        parameterfile << "\n";
    }
    parameterfile.close();
}


