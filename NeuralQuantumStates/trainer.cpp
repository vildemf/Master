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
    int nparameters = model.getNParameters();
    m_minimizer->setUp(nparameters);
    VectorXd shift(nparameters);

    if (m_writeIterativeExpectations) {
        m_iterativeExpectationsFile.open(m_IterativeExpectationsFilename, std::fstream::out);
    }

    for (int iteration=0; iteration<m_numberOfIterations; iteration++) {
        string foldername = "/Users/Vilde/Documents/masters/NQS_paper/tryHOrbm/";
        string filename = foldername + "MethodSelectionSampling/Training/blocking/" +
                "ISintPDEpoch" + std::to_string(iteration) + ".txt";
        //method.setWriteEnergiesForBlocking(filename);


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
    if (minimizertype=="adam") {
        unique_ptr<GradientDescent> minimizer(new GradientDescentADAM(learningrate));
        m_minimizer = move(minimizer);
    } else {
        unique_ptr<GradientDescent> minimizer(new GradientDescentSimple(learningrate, gamma));
        m_minimizer = move(minimizer);
    }
}

void Trainer::setWriteIterativeExpectations(string filename) {
    m_writeIterativeExpectations    = true;
    m_IterativeExpectationsFilename = filename;
}

void Trainer::printInfo(int iteration, double gradientNorm) {
    std::cout << "Gradient norm:         " << gradientNorm << std::endl
              << "Training epoch " << iteration << std::endl;
}
