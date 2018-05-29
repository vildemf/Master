#include "sampler.h"

using std::mt19937_64;

Sampler::Sampler(int seed) {
    m_randomEngine = mt19937_64(seed);
}
