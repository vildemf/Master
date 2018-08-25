#include "gradientdescentsimple.h"
#include "iostream"

using Eigen::VectorXd;

GradientDescentSimple::GradientDescentSimple(double learningrate, double gamma) : GradientDescent() {
    m_eta   = learningrate;
    m_gamma = gamma;
}

VectorXd GradientDescentSimple::computeParameterShift(const VectorXd &gradient, int nparameters, int iteration) {
    /*
     * The function computes the shift with which the network parameters should be updated according to the
     * simple gradient descent algoirthm.
     */

    VectorXd shift(nparameters);

    shift       = m_gamma*m_prevShift + m_eta*gradient;
    m_prevShift = shift;

    return -shift;
}


void GradientDescentSimple::setUp(int nparameters) {
    /*
     * Set up to be executed when the number of network parameters of the model is known.
     */

    m_prevShift.resize(nparameters);
}
