#ifndef TRAINER_H
#define TRAINER_H

#include "montecarlomethod.h"
#include "gradientdescent/gradientdescentsimple/gradientdescentsimple.h"
#include "gradientdescent/gradientdescentadam/gradientdescentadam.h"

class Trainer {
private:
    int                              m_numberOfIterations;
    std::unique_ptr<GradientDescent> m_minimizer;
    std::fstream                     m_iterativeExpectationsFile;
    bool                             m_writeIterativeExpectations;
    std::string                      m_IterativeExpectationsFilename;

    void initializeMinimizer(double learningrate, std::string minimizertype);

public:
    Trainer(int numberOfIterations, double learningrate, std::string minimizertype);

    void train(MonteCarloMethod &method, QuantumModel &model);
    void printInfo(int iteration, double gradientNorm);
    void setWriteIterativeExpectations(std::string filename);
};

#endif // TRAINER_H
