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


    NeuralQuantumState(double sigma, int nparticles, int nh, int ndim, std::string initialization, int seed);

    virtual double  computePsi(const Eigen::VectorXd &x, const Eigen::VectorXd &Q);
    double          computeLaplacian() const;
    Eigen::VectorXd computeParameterDerivative();

    double          quantumForce(int updateCoordinate);
    double          quantumForce(int updateCoordinate, const Eigen::VectorXd &x, const Eigen::VectorXd &sigmoidQ);

    void            setPsiComponents();
    void            setPsiComponents(const Eigen::VectorXd &Q);
    void            setPsiComponents(const Eigen::VectorXd &Q, const Eigen::VectorXd &sigmoidQ);

    void computeQ(const Eigen::VectorXd &x, Eigen::VectorXd &Q);
    void computeSigmoidQ(const Eigen::VectorXd &Q, Eigen::VectorXd &sigmoidQ);

    const Eigen::VectorXd& getParamterDerivative();
    void            setParameterDerivative(const Eigen::VectorXd &parameterDerivative);

    void            setInverseDistances();
    void            setInverseDistances(int particle);
    const Eigen::MatrixXd& getInverseDistances() const;

    int             getNParticles() const;
    int             getNX() const;
    int             getNH() const;
    int             getNDim() const;
    double          getPsi();
    const Eigen::VectorXd& getX() const;
    double          getA(int i);
    double          getB(int j);
    double          getW(int i, int j);
    void            setPsi(double psi);
    void            setX(const Eigen::VectorXd &x);
    void            setX(int i, double xi);
    void            setA(int i, double ai);
    void            setB(int j, double bj);
    void            setW(int i, int j, double wij);
};


#endif // NEURALQUANTUMSTATE_H
