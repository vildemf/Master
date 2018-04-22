#include <iostream>
#include "sampler/metropolis/metropolis.h"
#include "sampler/gibbs/gibbs.h"

using namespace std;

int main() {
    // Nqs parameters
    int nx = 4; // Number which represents particles*dimensions.
    int nh = 2; // Number of hidden units.
    int dim = 2; // Number of spatial dimensions
    double sigma = 1.0; // Normal distribution visibles
    bool gaussianInitialization = true; // Weights & biases (a,b,w) initialized uniformly or gaussian

    // Sampler parameters
    int nCycles = 300;  // 1000
    int nSamples = 10000;  // 100
    // Metropolis
    double step = 2.5;

    // Hamiltonian parameters
    double omega = 1.0;
    bool includeInteraction = true; // Include interaction or not

    // Optimizer parameters
    // SGD parameter
    double eta = 0.01; // SGD learning rate
    int nPar = nx + nh + nx*nh;

    // filename
    string filename = "/Users/Vilde/Documents/masters/NQS_paper/tryHOrbm/RBMoutput.txt";


    // Create objects for the sampler:
    Hamiltonian hamiltonian(omega, includeInteraction);
    NeuralQuantumState nqs(nh, nx, dim, sigma, gaussianInitialization);
    Sgd optimizer(eta, nPar);

    // Create the sampler:
    Metropolis metropolisSampler(nSamples, nCycles, step, hamiltonian, nqs, optimizer, filename);

    metropolisSampler.runOptimizationSampling();



    return 0;
}
