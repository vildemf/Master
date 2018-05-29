#ifndef NEURALQUANTUMSTATE_H
#define NEURALQUANTUMSTATE_H


#include <Eigen/Dense>
#include <random>

class NeuralQuantumState {
protected:
    double m_positiveDefiniteFactor;

    int             m_nx;
    int             m_nh;
    int             m_ndim;
    int             m_nparticles;
    double          m_sig;
    double          m_sig2;
    Eigen::VectorXd m_x;
    Eigen::VectorXd m_a;
    Eigen::VectorXd m_b;
    Eigen::MatrixXd m_w;
    // Quantities to store for computational efficiency
    double          m_psi;
    Eigen::VectorXd m_sigmoidQ;
    Eigen::VectorXd m_derSigmoidQ;
    Eigen::VectorXd m_parameterDerivative;
    Eigen::MatrixXd m_inverseDistances;

    void            initializeParametersFromfile(std::string filename);
    void            initializeParametersGaussian(std::mt19937_64 randomengine);
    void            initializePositions(std::mt19937_64 randomengine);

public:


    NeuralQuantumState(int nparticles, int nh, int ndim, std::string initialization, int seed);

    virtual double  computePsi(Eigen::VectorXd x, Eigen::VectorXd Q);
    double          computeLaplacian();
    Eigen::VectorXd computeParameterDerivative();

    double          quantumForce(int updateCoordinate);
    double          quantumForce(int updateCoordinate, Eigen::VectorXd x, Eigen::VectorXd sigmoidQ);


    void            setPsiComponents();
    void            setPsiComponents(Eigen::VectorXd Q);
    void            setPsiComponents(Eigen::VectorXd Q, Eigen::VectorXd sigmoidQ);

    Eigen::VectorXd computeQ(Eigen::VectorXd x);
    Eigen::VectorXd computeSigmoidQ(Eigen::VectorXd Q);

    Eigen::VectorXd getParamterDerivative();
    void            setParameterDerivative(Eigen::VectorXd parameterDerivative);

    void            setInverseDistances();
    void            setInverseDistances(int particle);
    Eigen::MatrixXd getInverseDistances();

    int             getNParticles();
    int             getNX();
    int             getNH();
    int             getNDim();
    double          getPsi();
    Eigen::VectorXd getX();
    double          getA(int i);
    double          getB(int j);
    double          getW(int i, int j);
    void            setPsi(double psi);
    void            setX(Eigen::VectorXd x);
    void            setX(int i, double xi);
    void            setA(int i, double ai);
    void            setB(int j, double bj);
    void            setW(int i, int j, double wij);
};


#endif // NEURALQUANTUMSTATE_H
