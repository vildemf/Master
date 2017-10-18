#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <cmath>

using namespace std;
using namespace Eigen;

int main()
{
    // INITIALIZATIONS
    int nx = 2;
    int nh = 4;
    int n_cycles = 7000;  // 1000
    int n_samples = 1000;  // 100
    double sigma = 1.0; // Normal distribution visibles
    double omega = 1.0;
    double eta = 4.0; // Learning rate
    double x_mean;  // Normal distribution visibles
    double der1lnPsi;
    double der2lnPsi;
    // ASGD parameters
    double a = 50.0;
    double A = 20.0;
    double t_prev = A;
    double t = A;
    double f_min = -0.5;
    double f_max = 2.0;
    double asgd_omega = 1.0;
    double asgd_X_prev;
    double f;



    // DECLARATIONS
    // The Boltzmann machine / wave function
    VectorXd x;
    VectorXd h;
    VectorXd b;
    VectorXd c;
    MatrixXd w;
    x.resize(nx);
    h.resize(nh);
    b.resize(nx);
    c.resize(nh);
    w.resize(nx, nh);
    // Wf derived wrt rbm parameters, to be added up for each sampling
    VectorXd derPsi_b;
    VectorXd derPsi_c;
    MatrixXd derPsi_w;
    VectorXd derPsi_b_temp;
    VectorXd derPsi_c_temp;
    MatrixXd derPsi_w_temp;
    derPsi_b.resize(nx);
    derPsi_c.resize(nh);
    derPsi_w.resize(nx, nh);
    derPsi_b_temp.resize(nx);
    derPsi_c_temp.resize(nh);
    derPsi_w_temp.resize(nx, nh);
    // Local energy times wf derived wrt rbm parameters, to be added up for each sampling
    VectorXd EderPsi_b;
    VectorXd EderPsi_c;
    MatrixXd EderPsi_w;
    EderPsi_b.resize(nx);
    EderPsi_c.resize(nh);
    EderPsi_w.resize(nx, nh);
    // Gradient, to be computed at each cycle, after all sampling iterations completed
    VectorXd grad_b;
    VectorXd grad_c;
    MatrixXd grad_w;
    VectorXd grad_b_prev;
    VectorXd grad_c_prev;
    MatrixXd grad_w_prev;
    grad_b.resize(nx);
    grad_c.resize(nh);
    grad_w.resize(nx, nh);
    grad_b_prev.resize(nx);
    grad_c_prev.resize(nh);
    grad_w_prev.resize(nx, nh);
    // Other
    VectorXd probHgivenX;
    VectorXd Q;
    probHgivenX.resize(nh);
    Q.resize(nh);

    // ASSIGNMENTS
    // The visible units/ positions
    std::uniform_real_distribution<double> distribution_initX(0,1);
    mt19937 rgen_Gibbs;
    std::random_device rd_Gibbs;
    rgen_Gibbs.seed(rd_Gibbs());
    for(int i=0; i<nx; i++){
        x(i)=distribution_initX(rgen_Gibbs);
    }
    // The rbm parameters
    int seed_initRBM=12345;
    float sigma_initRBM = 0.01;
    std::default_random_engine generator_initRBM(seed_initRBM);
    std::normal_distribution<double> distribution_initRBM(0,sigma_initRBM);
    for (int i=0; i<nx; i++){
        b(i) = distribution_initRBM(generator_initRBM);
    }
    for (int i=0; i<nh; i++){
        c(i) = distribution_initRBM(generator_initRBM);
    }
    for (int i=0; i<nx; i++){
        for (int j=0; j<nh; j++){
            w(i,j) = distribution_initRBM(generator_initRBM);
        }
    }

    // START
    // Minimization loop
    for (int cycles=0; cycles<n_cycles; cycles++) {
        // Variables to store summations during sampling
        double Eloc = 0;
        double Eloc_temp = 0;
        double Eloc2 = 0;
        derPsi_b.setZero();
        derPsi_c.setZero();
        derPsi_w.setZero();
        EderPsi_b.setZero();
        EderPsi_c.setZero();
        EderPsi_w.setZero();

        // Sampling loop
        for (int samples=0; samples<n_samples; samples++) {
            // Compute prob of hidden given visible and set hidden values accordingly
            std::uniform_real_distribution<double> distribution_setH(0,1);
            for (int j=0; j<nh; j++) {
                probHgivenX(j) = 1.0/(1 + exp(c(j) + x.transpose()*w.col(j)));
                h(j) = distribution_setH(rgen_Gibbs) < probHgivenX(j);
            }
            // Set new positions (visibles) given hidden, according to normal distribution
            default_random_engine generator_x;
            for (int i=0; i<nx; i++) {
                x_mean = b(i) + w.row(i)*h;
                normal_distribution<double> distribution_x(x_mean, sigma);
                //cout << cycles << "   " << samples << "   " << i << "   " << x_mean << endl;
                x(i) = distribution_x(generator_x);
            }
            if (samples > 0.1*n_samples) {
                // Compute the local energy
                Q = -c - (x.transpose()*w).transpose();
                Eloc_temp = 0;
                // Loop over the visibles (n_particles*n_coordinates) for the Laplacian
                for (int r=0; r<nx; r++) {
                    double sum1 = 0;
                    double sum2 = 0;
                    for (int j=0; j<nh; j++) {
                        sum1 += w(r,j)/(1.0+exp(-Q(j)));
                        sum2 += w(r,j)*w(r,j)*exp(Q(j))/((exp(Q(j))+1.0)*(exp(Q(j))+1.0));
                    }
                    der1lnPsi = -0.5*(x(r) - b(r)) - sum1;
                    der2lnPsi = -0.5 + sum2;
                    Eloc_temp += -der1lnPsi*der1lnPsi - der2lnPsi + omega*omega*x(r)*x(r);
                }
                Eloc_temp = 0.5*Eloc_temp;
                Eloc += Eloc_temp;

                // Compute the 1/psi * dPsi/dalpha_i, that is Psi derived wrt each RBM parameter.
                for (int i=0; i<nx; i++) {
                    derPsi_b_temp(i) = 0.5*(x(i) - b(i));
                }
                for (int j=0; j<nh; j++) {
                    derPsi_c_temp(j) = -1.0/(1.0+exp(-Q(j)));
                }
                for (int i=0; i<nx; i++) {
                    for (int j=0; j<nh; j++) {
                        derPsi_w_temp(i, j) = x(i)/(1.0+exp(Q(j)));
                    }
                }
                // Add up values for expectation values
                derPsi_b += derPsi_b_temp;
                derPsi_c += derPsi_c_temp;
                derPsi_w += derPsi_w_temp;
                EderPsi_b += Eloc_temp*derPsi_b_temp;
                EderPsi_c += Eloc_temp*derPsi_c_temp;
                EderPsi_w += Eloc_temp*derPsi_w_temp;
                Eloc2 += Eloc_temp*Eloc_temp;
            }
        }
        // Compute expectation values
        double n_samp = n_samples - 0.1*n_samples;
        Eloc = Eloc/n_samp;
        Eloc2 = Eloc2/n_samp;
        derPsi_b = derPsi_b/n_samp;
        derPsi_c = derPsi_c/n_samp;
        derPsi_w = derPsi_w/n_samp;
        EderPsi_b = EderPsi_b/n_samp;
        EderPsi_c = EderPsi_c/n_samp;
        EderPsi_w = EderPsi_w/n_samp;

        double variance = Eloc2 - Eloc*Eloc;

        // Compute gradient
        grad_b = 2*(EderPsi_b - Eloc*derPsi_b);
        grad_c = 2*(EderPsi_c - Eloc*derPsi_c);
        grad_w = 2*(EderPsi_w - Eloc*derPsi_w);


        // ASGD parameters
        f = f_min + (f_max - f_min)/(1 - (f_max/f_min)*exp(-asgd_X_prev/asgd_omega));
        t = 0;
        if (t < (t_prev + f)) t=t_prev+f;
        if (cycles==0 || cycles==1) t=20.0;
        double gamma = a/(t+A);
        //cout << asgd_X_prev << endl;

        // Compute new parameters
        for (int i=0; i<nx; i++) {
            // SGD
            //b(i) = b(i) - eta*grad_b(i);
            // ASGD
            b(i) = b(i) - gamma*grad_b(i);
        }
        for (int j=0; j<nh; j++) {
            // SGD
            //c(j) = c(j) - eta*grad_c(j);
            // ASGD
            c(j) = c(j) - gamma*grad_c(j);
        }
        for (int i=0; i<nx; i++) {
            for (int j=0; j<nh; j++) {
                // SGD
                //w(i,j) = w(i,j) - eta*grad_w(i,j);
                // ASGD
                w(i,j) = w(i,j) - gamma*grad_w(i,j);
            }
        }

        double gradnorm = sqrt(grad_b.squaredNorm() + grad_c.squaredNorm() + grad_w.squaredNorm());
        cout << cycles << "   " << Eloc << "   " << variance << "   " << b(0) << " " << b(1) << " " << c(0) << " " << c(1) << " " << c(2) << " " << c(3) << " " << w(0,0) << " " << w(0,1) << " " << w(0,2) << " " << w(0,3) << " " << w(1,0) << " " << w(1,1) << " " << w(1,2) << " " << w(1,3) << endl;

        // Prepare values needed for next round of ASGD
        double sum3 = 0;
        for (int i=0; i<nx; i++) {
            for (int j=0; j<nh; j++) {
                sum3 += grad_w(i,j)*grad_w_prev(i,j);
            }
        }

        asgd_X_prev = -grad_b.dot(grad_b_prev) - grad_c.dot(grad_c_prev) - sum3;
        grad_b_prev = grad_b;
        grad_c_prev = grad_c;
        grad_w_prev = grad_w;
        t_prev = t;


    }
    //cout << probHgivenV << endl;


}

