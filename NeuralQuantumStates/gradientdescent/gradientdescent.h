#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include <Eigen/Dense>

class GradientDescent {

public:
    GradientDescent();

    virtual Eigen::VectorXd computeParameterShift(const Eigen::VectorXd &gradient, int nparameters, int iteration) = 0;
    virtual void            setUp(int nparameters) = 0;
};

#endif // GRADIENTDESCENT_H
