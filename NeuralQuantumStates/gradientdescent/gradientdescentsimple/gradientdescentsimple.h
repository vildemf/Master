#ifndef GRADIENTDESCENTSIMPLE_H
#define GRADIENTDESCENTSIMPLE_H

#include "gradientdescent/gradientdescent.h"

class GradientDescentSimple : public GradientDescent {
private:
    double          m_eta;
    double          m_gamma;
    Eigen::VectorXd m_prevShift;

public:
    GradientDescentSimple(double learningrate);

    Eigen::VectorXd computeParameterShift(Eigen::VectorXd gradient, int nparameters, int iteration);
    void            setUp(int nparameters);
};

#endif // GRADIENTDESCENTSIMPLE_H
