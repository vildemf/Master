#include <iostream>
#include "trainer.h"

using namespace std;

int main()
{
    string foldername = "/Users/Vilde/Documents/masters/NQS_paper/tryHOrbm/";
    random_device rd;

    // Model
    double harmonicoscillatorOmega = 1.0;
    bool couloumbinteraction       = true;
    string nqsType                 = "general"; // positivedefinite, general
    string nqsInitialization       = "randomuniform"; // randomuniform, randomgaussian, filename of file to read from
    int nParticles                 = 2;
    int nDimensions                = 2;
    int nHidden                    = 2;
    int seed1                      = rd();

    shared_ptr<QuantumModel> model = make_shared<QuantumModel>(harmonicoscillatorOmega,
                                                               couloumbinteraction, nqsType, nqsInitialization,
                                                               nParticles, nDimensions, nHidden, seed1);

    // Method
    int numberOfSamples = 5e4;
    string samplertype  = "bruteforce"; // bruteforce, hastings (choose gibbs by alternative constructor)
    double step         = 2.5;
    int seed2           = rd();

    MonteCarloMethod method(numberOfSamples, model, seed2, samplertype, step);
    // Gibbs:
    //MonteCarloMethod method(numberOfSamples, model, rd());
    //method.runMonteCarlo();



    // Trainer
    int numberOfIterations = 300;
    double learningrate    = 0.02;
    string minimizertype   = "adam"; // simple, adam

    Trainer trainer(numberOfIterations, learningrate, minimizertype);
    trainer.train(method, *model);



    string paramfilename = foldername + "weights.txt";
    //model->writeParametersToFile(paramfilename);

    string blockingfilename = foldername + "newblocking.txt";
    //method.setWriteEnergiesForBlocking(blockingfilename);
    //method.runMonteCarlo();

    string trainerfilename = foldername + "trainingvalues.txt";
    //trainer.setWriteIterativeExpectations(trainerfilename);
    //trainer.train(method, *model);

    //model->writeParametersToFile(foldername + "trainedparameters.txt");



    return 0;
}
