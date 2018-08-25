#include "trainer.h"
#include "iostream"

using Eigen::VectorXd;
using std::string;
using std::unique_ptr;

Trainer::Trainer(int numberOfIterations, double learningrate, string minimizertype, double gamma) {

    m_numberOfIterations         = numberOfIterations;
    m_writeIterativeExpectations = false;

    initializeMinimizer(learningrate, minimizertype, gamma);

}


void Trainer::train(MonteCarloMethod &method, QuantumModel &model) {
    /*
     * The function trains the model using the given Monte Carlo method.
     */

    int nparameters = model.getNParameters();
    m_minimizer->setUp(nparameters);
    VectorXd shift(nparameters);

    if (m_writeIterativeExpectations) {
        m_iterativeExpectationsFile.open(m_IterativeExpectationsFilename, std::fstream::out);
    }

    for (int iteration=0; iteration<m_numberOfIterations; iteration++) {
        method.runMonteCarlo();
        printInfo(iteration, model.getGradientNorm());

        if (m_writeIterativeExpectations) {
            m_iterativeExpectationsFile << model.getExpectedLocalEnergy() << " "
                                        << model.getVariance() << " "
                                        << model.getGradientNorm() << "\n";
        }

        // This is where stopping criteria should be

        shift = m_minimizer->computeParameterShift(model.getGradient(), nparameters, iteration+1);
        model.shiftParameters(shift);
    }

    if (m_writeIterativeExpectations) {
        m_iterativeExpectationsFile.close();
    }


}


void Trainer::initializeMinimizer(double learningrate, string minimizertype, double gamma) {
    /*
     * The function initializes the minimizer to be used for training.
     */

    if (minimizertype=="adam") {
        unique_ptr<GradientDescent> minimizer(new GradientDescentADAM(learningrate));
        m_minimizer = move(minimizer);
    } else {
        unique_ptr<GradientDescent> minimizer(new GradientDescentSimple(learningrate, gamma));
        m_minimizer = move(minimizer);
    }
}

void Trainer::setWriteIterativeExpectations(string filename) {
    /*
     * Call this function to write the expected energy, (naive) variance and norm of the gradient to
     * file at each training epoch.
     *
     * Note: Should there also be a way for the user to turn this off?
     */

    m_writeIterativeExpectations    = true;
    m_IterativeExpectationsFilename = filename;
}

void Trainer::printInfo(int iteration, double gradientNorm) {
    /*
     * The function writes information to the terminal during training.
     */

    std::cout << "Gradient norm:         " << gradientNorm << std::endl
              << "Training epoch " << iteration << std::endl;
}
