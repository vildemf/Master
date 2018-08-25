#include "neuralquantumstate/nerualquantumstate.h"
#include "iostream"
#include <fstream>

using Eigen::VectorXd;
using Eigen::MatrixXd;

using std::string;
using std::mt19937_64;
using std::normal_distribution;
using std::uniform_real_distribution;

NeuralQuantumState::NeuralQuantumState(double sigma, int nparticles, int nh, int ndim, string initialization, int seed) {
    /*
     * The constructor of the Neural Quantum State (NQS) base class. It represents a wavefunction modeled by a restricted
     * Boltzmann machine. It is the general NQS which can represent both real and complex wavefunctions. The derived
     * class NeuralQuantumStatePositiveDefinite can only represent positive definite wavefunctions. The benefit of
     * it is that its position configurations can be sampled with the Gibbs method.
     *
     * Since many calculations are similar for the two except for one factor, the calculations are implemented
     * only once, in the base class, and the factor difference is implemented by the variable
     * m_positiveDefiniteFactor which is 1.0 for the general wavefunction and 0.5 for the positive definite one.
     *
     * This constructor initalizes class variables, including the Boltzmann machine's weights and biases, which are
     * initialized with uniform random numbers, Gaussian random numbers or numbers of an already trained network
     * read from file, depending on the user's choice. Finally, the constructor initializes the position vector
     * with uniform random numbers.
     */

    m_positiveDefiniteFactor = 1.0;
    m_nx                     = nparticles*ndim;
    m_nh                     = nh;
    m_ndim                   = ndim;
    m_nparticles             = nparticles;
    m_sig                    = sigma;
    m_sig2                   = m_sig*m_sig;

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




double NeuralQuantumState::computePsi(const VectorXd &x, const VectorXd &Q) {
    /*
     * The function computes the value of the wavefunction for a given position vector and
     * pre-computed value of Q.
     * It is used by Metropolis to
     * - Compute the value of the trial wavefunction (used at every sampling step)
     * - Compute the value of the current wavefunction (only used when initializing and
     *   at the beginning of a new training epoch when weights have been updated)
     */

    double factor1 = (x-m_a).dot(x-m_a);
    factor1        = exp(-factor1/(2.0*m_sig2));

    double factor2 = 1.0;
    for (int j=0; j<m_nh; j++) {
        factor2   *= (1 + exp(Q(j)));
    }

    return factor1*factor2;
}





double NeuralQuantumState::computeLaplacian() const {
    /*
     * The function computes the Laplacian of the wavefunction.
     */

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
    /*
     * The function computes 1/psi * dPsi/dalpha_i, that is, the wavefunction
     * differentiated with respect to each of the network parameters a_i, b_j and w_ij.
     */

    VectorXd dPsi(m_nx + m_nh + m_nx*m_nh);

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
    /*
     * The function computes the quantum force for the given coordinate.
     * It is used by Metropolis Importance Sampling to compute the quantum force of the current
     * state, since it will use this object's current value of the position vector and the factor Q.
     */

    double sum1 = m_sigmoidQ.dot(m_w.row(updateCoordinate));

    return 2*m_positiveDefiniteFactor*(-(m_x(updateCoordinate) - m_a(updateCoordinate))/m_sig2 + sum1/m_sig2);
}

double NeuralQuantumState::quantumForce(int updateCoordinate, const VectorXd &x, const VectorXd &sigmoidQ) {
    /*
     * The function computes the quantum force for the given coordinate and position vector.
     * It is used by Metropolis Importance Sampling to comptue the quantum force of the trial state,
     * since it does not use this object's current value of the position vector, but a given position
     * vector (and the corresponding pre-computed value of the factor Q).
     */

    double sum1 = sigmoidQ.dot(m_w.row(updateCoordinate));

    return 2*m_positiveDefiniteFactor*(-(x(updateCoordinate) - m_a(updateCoordinate))/m_sig2 + sum1/m_sig2);
}







void NeuralQuantumState::computeQ(const VectorXd &x, VectorXd &Q) {
    /*
     * The function computes the value of the vector Q for a given position vector x.
     */

    Q = m_b + (x.transpose()*m_w).transpose()/m_sig2;
}

void NeuralQuantumState::computeSigmoidQ(const VectorXd &Q, VectorXd &sigmoidQ) {
    /*
     * The function computes the value of sigmoidQ.
     */

    for (int j=0; j<m_nh; j++) {
        sigmoidQ(j) = 1./(1 + exp(-Q(j)));
    }
}


void NeuralQuantumState::setPsiComponents(const VectorXd &Q, const VectorXd &sigmoidQ) {
    /*
     * The point of the function setPsiComponents(...) is to store the values sigmoidQ and
     * derSigmoidQ which are used several times, to save computation time. After positions have
     * been updated by the sampler, one of the implementations of this function should be called to make
     * sure sigmoidQ and derSigmoidQ are also updated. Different implementations will be called depending
     * on whether the sampler has already computed some values, to avoid recalculations.
     *
     * This implementation of setPsiComponents(...) is used by Metropolis Importance Sampling, which
     * have already calculated sigmoidQ, hence it is given as an argument.
     */

    m_sigmoidQ           = sigmoidQ;
    double expQj;
    for (int j=0; j<m_nh; j++) {
        expQj            = exp(Q(j));
        m_derSigmoidQ(j) = expQj/((1+expQj)*(1+expQj));
    }
}
void NeuralQuantumState::setPsiComponents(const VectorXd &Q) {
    /*
     * The point of the function setPsiComponents(...) is to store the values sigmoidQ and
     * derSigmoidQ which are used several times, to save computation time. After positions have
     * been updated by the sampler, one of the implementations of this function should be called to make
     * sure sigmoidQ and derSigmoidQ are also updated. Different implementations will be called depending
     * on whether the sampler has already computed some values, to avoid recalculations.
     *
     * This implementation of setPsiComponents(...) is used by Brute Force, which
     * has neither precomputed sigmoidQ nor derSigmoidQ, hence both are computed here.
     */

    double expQj;

    for (int j=0; j<m_nh; j++) {
        expQj            = exp(Q(j));
        m_sigmoidQ(j)    = 1./(1 + exp(-Q(j)));
        m_derSigmoidQ(j) = expQj/((1+expQj)*(1+expQj));
    }
}
void NeuralQuantumState::setPsiComponents() {
    /*
     * The point of the function setPsiComponents(...) is to store the values sigmoidQ and
     * derSigmoidQ which are used several times, to save computation time. After positions have
     * been updated by the sampler, one of the implementations of this function should be called to make
     * sure sigmoidQ and derSigmoidQ are also updated. Different implementations will be called depending
     * on whether the sampler has already computed some values, to avoid recalculations.
     *
     * This implementation of setPsiComponents(...) is used by Gibbs, which
     * has neither precomputed sigmoidQ nor derSigmoidQ, hence both are computed here.
     * Additionally, it has not computed the new Q, so it also has to be computed here, rather than taken
     * as an argument.
     */

    VectorXd Q(m_nh);
    computeQ(m_x, Q);
    setPsiComponents(Q);
}




const VectorXd& NeuralQuantumState::getParamterDerivative() {
    return m_parameterDerivative;
}

int NeuralQuantumState::getNX() const {
    return m_nx;
}
int NeuralQuantumState::getNH() const {
    return m_nh;
}
int NeuralQuantumState::getNDim() const {
    return m_ndim;
}

int NeuralQuantumState::getNParticles() const {
    return m_nparticles;
}

double NeuralQuantumState::getPsi() {
    return m_psi;
}

const VectorXd& NeuralQuantumState::getX() const {
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


void NeuralQuantumState::setParameterDerivative(const VectorXd &parameterDerivative) {
    m_parameterDerivative = parameterDerivative;
}

void NeuralQuantumState::setPsi(double psi) {
    m_psi = psi;
}

void NeuralQuantumState::setX(const VectorXd &x) {
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

const MatrixXd& NeuralQuantumState::getInverseDistances() const {
    return m_inverseDistances;
}




void NeuralQuantumState::setInverseDistances(int particle) {
    /*
     * The setInverseDistances(...) function updates the matrix m_inverseDistances, is an upper triangular
     * matrix which stores 1/r_ij as its elements, where r_ij = the distance between particle i and j.
     * It should be called whenever the position vector m_x has been updated.
     *
     * This setInverseDistances(...) should be called when just a signle particle has been moved. It
     * takes the index of the moved particle as its argument.
     */

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
    /*
     * The setInverseDistances(...) function updates the matrix m_inverseDistances, is an upper triangular
     * matrix which stores 1/r_ij as its elements, where r_ij = the distance between particle i and j.
     * It should be called whenever the position vector m_x has been updated.
     *
     * This implementation of setInverseDistances(...) should be called when all particles have been moved.
     */

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
    /*
     * The function initializes the network parameters with Gaussian random numbers. It is called by
     * the constructor if Gaussian numbers have been chosen by the user.
     */

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
    /*
     * The function initializes the network parameters with already trained parameters read from file. It is
     * called by the constructor if the user has chosen this initialization option and given a file name.
     * The file is assumed to be on the following format:
     * a0 a1 a2 ...
     * b0 b1 b2 ...
     * w00 w01 w02 ...
     * w10 w11 w12 ...
     * ...
     *
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
    /*
     * The function initializes the positions of the particles with uniform random numbers.
     */

    m_x.resize(m_nx);
    uniform_real_distribution<double> distribution_initX(-0.5,0.5);

    for(int i=0; i<m_nx; i++){
        m_x(i) = distribution_initX(randomengine);
    }

    // Creates a strictly upper triangular matrix
    setInverseDistances();
}
