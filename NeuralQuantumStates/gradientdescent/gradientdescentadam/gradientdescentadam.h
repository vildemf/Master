#ifndef GRADIENTDESCENTADAM_H
#define GRADIENTDESCENTADAM_H

#include "gradientdescent/gradientdescent.h"

class GradientDescentADAM : public GradientDescent {
private:
    double          m_eta; // learning rate, typically 1e-3
    double          m_epsilon; // small regularization constant to prevent divergencies, typically 1e-8
    double          m_beta1; // the memory of the lifetime of the first moment of the gradient, typically 0.9
    double          m_beta2; // -||- second moment of the gradient, typically 0.99

    Eigen::VectorXd m_mprev;
    Eigen::VectorXd m_sprev;

public:
    GradientDescentADAM(double learningrate);

    Eigen::VectorXd computeParameterShift(Eigen::VectorXd gradient, int nparameters, int iteration);
    void            setUp(int nparameters);
};

#endif // GRADIENTDESCENTADAM_H


