#include "hamiltonian.h"

Hamiltonian::Hamiltonian() {

}

void Hamiltonian::updateCumulativeLocalEnergy(double &Eloc) {
    Q = b + (1.0/sig2)*(x.transpose()*w).transpose();
    Eloc_temp = 0;
    // Loop over the visibles (n_particles*n_coordinates) for the Laplacian
    for (int r=0; r<nx; r++) {
        double sum1 = 0;
        double sum2 = 0;
        for (int j=0; j<nh; j++) {
            sum1 += w(r,j)/(1.0+exp(-Q(j)));
            sum2 += w(r,j)*w(r,j)*exp(Q(j))/((exp(Q(j))+1.0)*(exp(Q(j))+1.0));
        }
        der1lnPsi = -(x(r) - a(r))/sig2 + sum1/sig2;
        der2lnPsi = -1.0/sig2 + sum2/(sig2*sig2);
        Eloc_temp += -der1lnPsi*der1lnPsi - der2lnPsi + omega*omega*x(r)*x(r);


    }
    Eloc_temp = 0.5*Eloc_temp;

    // With interaction:
    Eloc_temp += interaction(x, nx, dim);


    Eloc += Eloc_temp;
}
