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

    // Sampler parameters
    int nCycles = 15000;  // 1000
    int nSamples = 10000;  // 100
    // Metropolis
    double step = 1.0;

    // Hamiltonian parameters
    double omega = 1.0;

    // Optimizer parameters
    // SGD parameter
    double eta = 0.01; // SGD learning rate
    int nPar = nx + nh + nx*nh;
    // ASGD parameters
    double A = 20.0;
    double t_prev = A;
    double t = A;
    double asgd_X_prev;

    //random_device seedGenerator;


    // Create objects for the sampler:
    Hamiltonian hamiltonian(omega);
    NeuralQuantumState nqs(nh, nx, dim, sigma);
    Sgd optimizer(eta, nPar);

    // Create the sampler:
    Metropolis metropolisSampler(nSamples, nCycles, step, hamiltonian, nqs, optimizer);

    metropolisSampler.runOptimizationSampling();



    return 0;
}
