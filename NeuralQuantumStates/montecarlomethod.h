#ifndef MONTECARLOMETHOD_H
#define MONTECARLOMETHOD_H

#include "quantummodel.h"
#include <fstream>

class MonteCarloMethod {
private:
    int                           m_numberOfSamples;
    std::shared_ptr<QuantumModel> m_model;
    std::fstream                  m_energiesblockingfile;
    bool                          m_writeEnergiesForBlocking;
    std::string                   m_energiesblockingfilename;

public:
    MonteCarloMethod(int numberOfSamples, std::shared_ptr<QuantumModel> model, int seed);
    MonteCarloMethod(int numberOfSamples, std::shared_ptr<QuantumModel> model, int seed,
                     std::string samplertype, double step);

    void runMonteCarlo();
    void setWriteEnergiesForBlocking(std::string filename);

};

#endif // MONTECARLOMETHOD_H
