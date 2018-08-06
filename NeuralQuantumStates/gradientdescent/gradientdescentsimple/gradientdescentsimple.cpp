#include "gradientdescentsimple.h"
#include "iostream"

using Eigen::VectorXd;

GradientDescentSimple::GradientDescentSimple(double learningrate, double gamma) : GradientDescent() {
    m_eta   = learningrate;
    m_gamma = gamma;
}

VectorXd GradientDescentSimple::computeParameterShift(const VectorXd &gradient, int nparameters, int iteration) {
    VectorXd shift(nparameters);

    shift       = m_gamma*m_prevShift + m_eta*gradient;
    m_prevShift = shift;

    return -shift;
}


void GradientDescentSimple::setUp(int nparameters) {
    m_prevShift.resize(nparameters);
}
