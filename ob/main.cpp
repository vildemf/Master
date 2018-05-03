#include <iostream>
#include "sampler/metropolisimportancesampling/metropolisimportancesampling.h"
#include "sampler/metropolisbruteforce/metropolisbruteforce.h"
#include "sampler/gibbs/gibbs.h"

using namespace std;

int main() {
    // filenames
    string filename = "/Users/Vilde/Documents/masters/NQS_paper/tryHOrbm/RBMoutput.txt";
    string blockFilename = "/Users/Vilde/Documents/masters/NQS_paper/tryHOrbm/blocking.txt";

    // Nqs parameters
    int nParticles = 2;                   // Number of particles
    int nHidden = 2;                      // Number of hidden units.
    int nDim = 2;                         // Number of spatial dimensions
    int nVisible = nParticles*nDim;
    double sigma = 1.0;                   // Normal distribution visibles
    bool gaussianInitialization = true;   // Weights & biases (a,b,w) initialized uniformly or gaussian

    // Sampler parameters
    string samplemethod = "importance";   // Choose between "importance", "bruteforce" and "gibbs"
    int nCycles = 1000;                   // Number of optimization iterations
    int nSamples = 10000;           // Number of samples in each iteration
    random_device rd;                    // Seed
    // Metropolis
    double step = 0.45;

    // Hamiltonian parameters
    double omega = 1.0;
    bool includeInteraction = false;      // Include interaction or not



    // Optimizer parameters (choose either stochastic gradient descent (SGD) or adaptive SGD (ASGD))
    int nPar = nVisible + nHidden + nVisible*nHidden;
    string optimization = "sgd";        // choose between "sgd", "asgd" and "asgdWithOptionals"
    // SGD parameters
    double eta = 0.2;                   // must be >0. SGD learning rate (lr)

    // ASGD parameters. lr: gamma_i=a/(A+t_i) where t[i]=max(0, t[i-1]+f(-grad[i]*grad[i-1]))
    double a = 0.01;                     // must be >0. Proportional to the lr
    double A = 20.0;                     // must be >= 1. Inverse prop to the lr. (a/A) defines the max lr.
    // ASGD optional: parameters to the function f
    double asgdOmega;                    // must be >0. As omega->0, f-> step function.
    double fmax;                         // must be >0
    double fmin;                         // must be <0
    // ASGD optional: initial conditions
    double t0;                           // Suggested choices are t0=t1=A=20 (default)
    double t1;                           // or t0=t1=0




    // Starting the run based on user options

    // Create objects for the sampler:
    Hamiltonian hamiltonian(omega, includeInteraction);
    NeuralQuantumState nqs(nHidden, nVisible, nDim, sigma, gaussianInitialization);

    Sgd optimizer(eta, nPar);
    if (optimization=="sgd") {
        Sgd optimizer(eta, nPar);
    } else if (optimization=="asgd") {
        Asgd optimizer(a, A, nPar);
    } else if (optimization=="asgdWithOptionals") {
        Asgd optimizer(a, A, asgdOmega, fmax, fmin, t0, t1, nPar);
    } else {
        cout << "Error: Please choose one of the specified optimizers. Now running default gradient descent." << endl;
    }

    // Create the sampler:
    if (samplemethod=="importance") {
        MetropolisImportanceSampling metropolisSampler(nSamples, nCycles, step, hamiltonian, nqs, optimizer, filename,
                                     blockFilename, rd());
        metropolisSampler.runOptimizationSampling();
    } else if (samplemethod=="bruteforce") {
        MetropolisBruteForce metropolisSampler(nSamples, nCycles, step, hamiltonian, nqs, optimizer, filename,
                                     blockFilename, rd());
        metropolisSampler.runOptimizationSampling();
    } else if (samplemethod=="gibbs") {
        Gibbs gibbsSampler(nSamples, nCycles, hamiltonian, nqs, optimizer, filename, blockFilename, rd());
        gibbsSampler.runOptimizationSampling();
    } else {
        cout << "Error: Please choose one of the specified samplers.";
    }



    return 0;
}
