#include <iostream>
#include "trainer.h"
#include <math.h>
#include <chrono>

using namespace std;

int main()
{
    random_device rd;

    // Model
    double harmonicoscillatorOmega = 1.0;
    bool coulombinteraction       = true;
    string nqsType                 = "positivedefinite";
    string nqsInitialization       = "randomgaussian";
    int nParticles                 = 2;
    int nDimensions                = 2;
    int nHidden                    = 2;
    int seed1                      = rd();

    shared_ptr<QuantumModel> model = make_shared<QuantumModel>
                (harmonicoscillatorOmega,
                    coulombinteraction,
                    nqsType,
                    nqsInitialization,
                    nParticles,
                    nDimensions,
                    nHidden,
                    seed1);


    // Method
    int numberOfSamples = 1e5;
    string samplertype  = "importancesampling";
    double step         = 0.4;
    int seed2           = rd();

    MonteCarloMethod method(numberOfSamples, model, seed2, samplertype, step);


    // Trainer
    int numberOfIterations = 500;
    double learningrate    = 0.02;
    double gamma           = 0.05;
    string minimizertype   = "simple";

    Trainer trainer(numberOfIterations, learningrate, minimizertype, gamma);
    trainer.train(method, *model);


    return 0;
}
