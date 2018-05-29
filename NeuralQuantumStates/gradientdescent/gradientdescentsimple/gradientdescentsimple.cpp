#include "gradientdescentsimple.h"

using Eigen::VectorXd;

GradientDescentSimple::GradientDescentSimple(double learningrate) : GradientDescent() {
    m_eta   = learningrate;
    m_gamma = 0.0;
}

VectorXd GradientDescentSimple::computeParameterShift(VectorXd gradient, int nparameters, int iteration) {
    VectorXd shift(nparameters);

    shift       = m_gamma*m_prevShift + m_eta*gradient;
    m_prevShift = shift;

    return -shift;
}


void GradientDescentSimple::setUp(int nparameters) {
    m_prevShift.resize(nparameters);
}
