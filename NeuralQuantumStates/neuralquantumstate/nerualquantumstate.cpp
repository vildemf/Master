#include "neuralquantumstate/nerualquantumstate.h"
#include "iostream"
#include <fstream>

using Eigen::VectorXd;
using Eigen::MatrixXd;

using std::string;
using std::mt19937_64;
using std::normal_distribution;
using std::uniform_real_distribution;

NeuralQuantumState::NeuralQuantumState(int nparticles, int nh, int ndim, string initialization, int seed) {
    m_positiveDefiniteFactor = 1.0; // changed to 0.5 if positive definite, in the derived constructor
    m_nx                     = nparticles*ndim;
    m_nh                     = nh;
    m_ndim                   = ndim;
    m_nparticles             = nparticles;
    m_sig                    = 1.0; //sigma;
    m_sig2                   = m_sig*m_sig;    //sigma*sigma;

    m_sigmoidQ           .resize(m_nh);
    m_derSigmoidQ        .resize(m_nh);
    m_parameterDerivative.resize(m_nx + m_nh + m_nx*m_nh);
    m_inverseDistances   .resize(m_nparticles, m_nparticles);
    m_inverseDistances   .setZero();

    mt19937_64 randomengine(seed);

    if (initialization=="randomgaussian") {
        initializeParametersGaussian(randomengine);
    } else if (initialization=="randomuniform"){
        m_a = VectorXd::Random(m_nx);
        m_b = VectorXd::Random(m_nh);
        m_w = MatrixXd::Random(m_nx, m_nh);
    } else {
        initializeParametersFromfile(initialization);
    }

    initializePositions(randomengine);

}




double NeuralQuantumState::computePsi(VectorXd x, VectorXd Q) {
    // Used by Metropolis
    // Computes the trial Psi - used at every sampling
    // Computes the current Psi - only used when initializing at
    // the beginning of a new cycle
    double factor1 = (x-m_a).dot(x-m_a);
    factor1        = exp(-factor1/(2.0*m_sig2));

    double factor2 = 1.0;
    for (int j=0; j<m_nh; j++) {
        factor2   *= (1 + exp(Q(j)));
    }

    return factor1*factor2;
}





double NeuralQuantumState::computeLaplacian() {
    double d1lnPsi;
    double d2lnPsi;
    double laplacian = 0.;
    double sumterm   = 0;

    // Loop over the visibles (n_particles*n_coordinates) for the Laplacian
    for (int i=0; i<m_nx; i++) {
        d1lnPsi      = -(m_x(i) - m_a(i))/m_sig2 + m_w.row(i).dot(m_sigmoidQ)/m_sig2;

        sumterm      = 0;
        for (int j=0; j<m_nh; j++) {
            sumterm += m_w(i,j)*m_w(i,j)*m_derSigmoidQ(j);
        }
        d2lnPsi      = -1.0/m_sig2 + sumterm/(m_sig2*m_sig2);

        d1lnPsi     *= m_positiveDefiniteFactor;
        d2lnPsi     *= m_positiveDefiniteFactor;

        laplacian   += -d1lnPsi*d1lnPsi - d2lnPsi;
    }
    return laplacian;
}


VectorXd NeuralQuantumState::computeParameterDerivative() {
    // Compute the 1/psi * dPsi/dalpha_i, that is Psi derived wrt each RBM parameter.
    VectorXd dPsi;
    dPsi.resize(m_nx + m_nh + m_nx*m_nh);

    for (int k=0; k<m_nx; k++) {
        dPsi(k) = (m_x(k) - m_a(k))/m_sig2;
    }
    for (int k=m_nx; k<(m_nx+m_nh); k++) {
        dPsi(k) = m_sigmoidQ(k-m_nx);
    }
    int k=m_nx + m_nh;
    for (int i=0; i<m_nx; i++) {
        for (int j=0; j<m_nh; j++) {
            dPsi(k) = m_x(i)*m_sigmoidQ(j)/m_sig2;
            k++;
        }
    }

    return m_positiveDefiniteFactor*dPsi;
}



double NeuralQuantumState::quantumForce(int updateCoordinate) {
    // Calculates the quantum force for the given coordinate for the current state
    double sum1 = m_sigmoidQ.dot(m_w.row(updateCoordinate));

    return 2*m_positiveDefiniteFactor*(-(m_x(updateCoordinate) - m_a(updateCoordinate))/m_sig2 + sum1/m_sig2);
}

double NeuralQuantumState::quantumForce(int updateCoordinate, VectorXd x, VectorXd sigmoidQ) {
    // Calculates the quantum force for the given coordinate for the trial state
    double sum1 = sigmoidQ.dot(m_w.row(updateCoordinate));

    return 2*m_positiveDefiniteFactor*(-(x(updateCoordinate) - m_a(updateCoordinate))/m_sig2 + sum1/m_sig2);
}







VectorXd NeuralQuantumState::computeQ(VectorXd x) {
    VectorXd Q = m_b + (x.transpose()*m_w).transpose()/m_sig2;

    return Q;
}

VectorXd NeuralQuantumState::computeSigmoidQ(VectorXd Q) {
    VectorXd sigmoidQ(m_nh);

    for (int j=0; j<m_nh; j++) {
        sigmoidQ(j) = 1./(1 + exp(-Q(j)));
    }
    return sigmoidQ;
}


void NeuralQuantumState::setPsiComponents(VectorXd Q, VectorXd sigmoidQ) {
    // To be called after position has been updated - used by Metropolis Importance Sampling,
    // which have
    // already computed some quantities for trial expressions.
    m_sigmoidQ           = sigmoidQ;
    double expQj;
    for (int j=0; j<m_nh; j++) {
        expQj            = exp(Q(j));
        m_derSigmoidQ(j) = expQj/((1+expQj)*(1+expQj));
    }
}
void NeuralQuantumState::setPsiComponents(VectorXd Q) {
    // To be called after position has been updated - used by Metropolis Brute Force
    double expQj;

    for (int j=0; j<m_nh; j++) {
        expQj            = exp(Q(j));
        m_sigmoidQ(j)    = 1./(1 + exp(-Q(j)));
        m_derSigmoidQ(j) = expQj/((1+expQj)*(1+expQj));
    }
}
void NeuralQuantumState::setPsiComponents() {
    // To be called after position has been updated - used by Gibbs
    setPsiComponents(computeQ(m_x));
}




VectorXd NeuralQuantumState::getParamterDerivative() {
    return m_parameterDerivative;
}

int NeuralQuantumState::getNX() {
    return m_nx;
}
int NeuralQuantumState::getNH() {
    return m_nh;
}
int NeuralQuantumState::getNDim() {
    return m_ndim;
}

double NeuralQuantumState::getPsi() {
    return m_psi;
}

VectorXd NeuralQuantumState::getX() {
    return m_x;
}
double NeuralQuantumState::getA(int i) {
    return m_a(i);
}
double NeuralQuantumState::getB(int j) {
    return m_b(j);
}
double NeuralQuantumState::getW(int i, int j) {
    return m_w(i,j);
}

int NeuralQuantumState::getNParticles() {
    return m_nparticles;
}

void NeuralQuantumState::setParameterDerivative(VectorXd parameterDerivative) {
    m_parameterDerivative = parameterDerivative;
}

void NeuralQuantumState::setPsi(double psi) {
    m_psi = psi;
}

void NeuralQuantumState::setX(VectorXd x) {
    m_x = x;
}
void NeuralQuantumState::setX(int i, double xi) {
    m_x(i) = xi;
}
void NeuralQuantumState::setA(int i, double ai) {
    m_a(i) = ai;
}
void NeuralQuantumState::setB(int j, double bj) {
    m_b(j) = bj;
}
void NeuralQuantumState::setW(int i, int j, double wij) {
    m_w(i,j) = wij;
}

MatrixXd NeuralQuantumState::getInverseDistances() {
    return m_inverseDistances;
}






void NeuralQuantumState::setInverseDistances(int particle) {
    // Call after m_x has been changed by one particle
    double distanceSquared;
    // Loop over all the other particles and change the distance to them
    int p=0;
    for (int i=0; i<=m_nx-m_ndim; i+=m_ndim) {

        distanceSquared=0;
        for (int d=0; d<m_ndim; d++) {
            distanceSquared += (m_x(particle*m_ndim+d) - m_x(i+d))*(m_x(particle*m_ndim+d) - m_x(i+d));
        }
        // Make sure to work with the upper part of the triangular matrix and ignore if p=particle
        if (particle < p) {
            m_inverseDistances(particle,p) = 1.0/sqrt(distanceSquared);
        } else if (particle > p) {
             m_inverseDistances(p,particle) = 1.0/sqrt(distanceSquared);
        }
        p++;
    }

}

void NeuralQuantumState::setInverseDistances() {
    // Call if all of m_x has been changed
    double distanceSquared;
    int p1 = 0;
    int p2;

    // Loop over each particle
    for (int i1=0; i1<m_nx-m_ndim; i1+=m_ndim) {
        p2 = p1+1;
        // Loop over each particle s that particle r hasn't been paired with
        for (int i2=(i1+m_ndim); i2<m_nx; i2+=m_ndim) {

            distanceSquared=0;
            // Loop over dimensions
            for (int d=0; d<m_ndim; d++) {
                distanceSquared += (m_x(i1+d) - m_x(i2+d))*(m_x(i1+d) - m_x(i2+d));
            }
            m_inverseDistances(p1,p2) = 1.0/sqrt(distanceSquared);
            p2++;
        }
        p1++;
    }
}



void NeuralQuantumState::initializeParametersGaussian(mt19937_64 randomengine) {
    double sigma_initRBM = 0.1;
    m_x.resize(m_nx);
    m_a.resize(m_nx);
    m_b.resize(m_nh);
    m_w.resize(m_nx, m_nh);
    normal_distribution<double> distributionParams(0,sigma_initRBM);

    for (int i=0; i<m_nx; i++){
        m_a(i) = distributionParams(randomengine);
    }
    for (int j=0; j<m_nh; j++){
        m_b(j) = distributionParams(randomengine);
    }
    for (int i=0; i<m_nx; i++){
        for (int j=0; j<m_nh; j++){
            m_w(i,j) = distributionParams(randomengine);
        }
    }
}
void NeuralQuantumState::initializeParametersFromfile(string filename) {
    /* this should not require the user to set nx and nh.
     Assumed format of file:
     a0 a1 a2 ...
     b0 b1 b2 ...
     w00 w01 w02 ...
     w10 w11 w12 ...
     ...
    */
    m_a.resize(m_nx);
    m_b.resize(m_nh);
    m_w.resize(m_nx, m_nh);

    std::ifstream input;
    input.open(filename, std::ifstream::in);
    double d;
    string line;
    int linecount = 0;
    while (std::getline(input, line)) {
        std::istringstream iss(line);
        if (linecount==0) {
            int i=0;
            while(iss >> d) {
                m_a(i) = d;
                i++;
            }
        } else if (linecount==1) {
            int j=0;
            while(iss >> d) {
                m_b(j) = d;
                j++;
            }
        } else {
            int j=0;
            while(iss >> d) {
                m_w(linecount-2, j) = d;
                j++;
            }
        }
        linecount++;
    }
}


void NeuralQuantumState::initializePositions(mt19937_64 randomengine) {
    m_x.resize(m_nx);
    uniform_real_distribution<double> distribution_initX(-0.5,0.5);

    for(int i=0; i<m_nx; i++){
        m_x(i) = distribution_initX(randomengine);
    }

    // Creates a strictly upper triangular matrix
    setInverseDistances();
}
