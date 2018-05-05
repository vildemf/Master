#include "neuralquantumstate.h"
#include <random>


NeuralQuantumState::NeuralQuantumState(int nh, int nx, int dim, double sigma, bool gaussianInitialization, int seed) {
    m_randomEngine = std::mt19937_64(seed);
    setup(nh, nx, dim, sigma, gaussianInitialization);
}

void NeuralQuantumState::setup(int nh, int nx, int dim, double sigma, bool gaussianInitialization) {
    m_nx = nx;
    m_nh = nh;
    m_dim  = dim;
    m_sig = sigma;
    m_sig2 = sigma*sigma;
    m_x.resize(m_nx); // positions/visibles
    m_h.resize(m_nh); // hidden
    m_a.resize(m_nx); // visible bias
    m_b.resize(m_nh); // hidden bias
    m_w.resize(m_nx, m_nh); // weights

    m_sigmoidQ.resize(m_nh);
    m_derSigmoidQ.resize(m_nh);

    if (gaussianInitialization) {
        setupWeights();
    } else {
        m_x = Eigen::VectorXd::Random(nx);
        m_a = Eigen::VectorXd::Random(nx);
        m_b = Eigen::VectorXd::Random(nh);
        m_w = Eigen::MatrixXd::Random(nx, nh);
    }

    setupPositions();
}

void NeuralQuantumState::setupWeights() {
    float sigma_initRBM = 0.001;
    std::normal_distribution<double> distribution_initRBM(0,sigma_initRBM);
    for (int i=0; i<m_nx; i++){
        m_a(i) = distribution_initRBM(m_randomEngine);
        //outfile << a(i) << " ";
    }
    for (int i=0; i<m_nh; i++){
        m_b(i) = distribution_initRBM(m_randomEngine);
        //outfile << b(i) << " ";
    }
    for (int i=0; i<m_nx; i++){
        for (int j=0; j<m_nh; j++){
            m_w(i,j) = distribution_initRBM(m_randomEngine);
            //outfile << w(i,j) << " ";
        }
    }
    //outfile << '\n';
}

void NeuralQuantumState::setupPositions() {
    std::uniform_real_distribution<double> distribution_initX(-0.5,0.5);
    for(int i=0; i<m_nx; i++){
        m_x(i)=distribution_initX(m_randomEngine);
    }
}



double NeuralQuantumState::computePsi(Eigen::VectorXd x, Eigen::VectorXd Q) {
    // Used by Metropolis
    // Computes the trial Psi - used at every sampling
    // Computes the current Psi - only used when initializing at
    // the beginning of a new cycle
    double psiFactor = (x-m_a).dot(x-m_a);
    psiFactor = exp(-psiFactor/(2.0*m_sig2));

    double factorExpQ = 1.0;
    for (int j=0; j<m_nh; j++) {
        factorExpQ *= (1 + exp(Q(j)));
    }
    return psiFactor*factorExpQ;
}





double NeuralQuantumState::quantumForce(int updateCoordinate) {
    // Calculates the quantum force for the given coordinate for the current state   
    double sum1 = m_sigmoidQ.dot(m_w.row(updateCoordinate));
    double Fcurrent = 2*(-(m_x(updateCoordinate) - m_a(updateCoordinate))/m_sig2 + sum1/m_sig2);
    return Fcurrent;
}

double NeuralQuantumState::quantumForce(int updateCoordinate, Eigen::VectorXd xTrial, Eigen::VectorXd Q, Eigen::VectorXd &sigmoidQ) {
    // Calculates the quantum force for the given coordinate for the trial state
    for (int j=0; j<m_nh; j++) {
        sigmoidQ(j) = 1./(1 + exp(-Q(j)));
    }
    double sum1 = sigmoidQ.dot(m_w.row(updateCoordinate));
    double Ftrial = 2*(-(xTrial(updateCoordinate) - m_a(updateCoordinate))/m_sig2 + sum1/m_sig2);
    return Ftrial;
}


void NeuralQuantumState::updatePsiComponents() {
    // To be called after position has been updated - used by Gibbs
    Eigen::VectorXd Q = m_b + (m_x.transpose()*m_w).transpose()/m_sig2;
    double expQj;
    double expNegQj;
    for (int j=0; j<m_nh; j++) {
        expQj = exp(Q(j));
        expNegQj = exp(-Q(j));
        m_sigmoidQ(j) = 1./(1 + expNegQj);
        m_derSigmoidQ(j) = expQj/((1+expQj)*(1+expQj));
    }
}

void NeuralQuantumState::updatePsiComponents(Eigen::VectorXd Q) {
    // To be called after position has been updated - used by Metropolis Brute Force
    double expQj;
    double expNegQj;
    for (int j=0; j<m_nh; j++) {
        expQj = exp(Q(j));
        expNegQj = exp(-Q(j));
        m_sigmoidQ(j) = 1./(1 + expNegQj);
        m_derSigmoidQ(j) = expQj/((1+expQj)*(1+expQj));
    }
}

void NeuralQuantumState::updatePsiComponents(Eigen::VectorXd Q, Eigen::VectorXd sigmoidQ) {
    // To be called after position has been updated - used by Metropolis Importance Sampling,
    // which have
    // already computed some quantities for trial expressions.
    m_sigmoidQ = sigmoidQ;

    double expQj;
    for (int j=0; j<m_nh; j++) {
        expQj = exp(Q(j));
        m_derSigmoidQ(j) = expQj/((1+expQj)*(1+expQj));
    }
}



