#include "gradientdescentadam.h"
#include "iostream"

using Eigen::VectorXd;

GradientDescentADAM::GradientDescentADAM(double learningrate) : GradientDescent() {
    m_eta     = learningrate;
    m_epsilon = 1e-8;
    m_beta1   = 0.9;
    m_beta2   = 0.99;
}

VectorXd GradientDescentADAM::computeParameterShift(Eigen::VectorXd gradient, int nparameters, int iteration) {
    VectorXd shift(nparameters);
    VectorXd m(nparameters);
    VectorXd s(nparameters);
    VectorXd gradientSquared(nparameters);

    m = m_beta1*m_mprev + (1-m_beta1)*gradient;

    for (int i=0; i<nparameters; i++) {
        gradientSquared(i) = gradient(i)*gradient(i);
    }
    s = m_beta2*m_sprev + (1-m_beta2)*gradientSquared;

    m_mprev = m;
    m_sprev = s;

    m = m/(1-pow(m_beta1, iteration));
    s = s/(1-pow(m_beta2, iteration));

    for (int i=0; i<nparameters; i++) {
        shift(i) = m(i)/(sqrt(s(i)) + m_epsilon);
    }

    return -m_eta*shift;;
}

void GradientDescentADAM::setUp(int nparameters) {
    m_mprev.resize(nparameters);
    m_sprev.resize(nparameters);

    m_mprev.setZero();
    m_sprev.setZero();
}
