// Created by Alessandro Maraio on 04/02/2020.
// Copyright (c) 2020 University of Sussex.
// contributor: Alessandro Maraio <am963@sussex.ac.uk>

#include <iostream>
#include <vector>
#include <cmath>
#include "spline1d.h"
#include "cuba.h"

extern "C" {


using DataType = double;

transport::spline1d<DataType> twopf_spline;
transport::spline1d<DataType> transfer_spline;

void set_twopf_data(double *k, DataType *data, int length) {
    std::vector<double> twopf_k;
    twopf_k.assign(k, k + length);

    std::vector<DataType> twopf_data;
    twopf_data.assign(data, data + length);

    twopf_spline.set_spline_data(twopf_k, twopf_data);
}


DataType get_twopf_data(DataType k) {
    return twopf_spline(k);
}


void set_transfer_data(double *k, DataType *data, int length) {
    std::vector<double> transfer_k;
    transfer_k.assign(k, k + length);

    std::vector<DataType> transfer_data;
    transfer_data.assign(data, data + length);

    transfer_spline.set_spline_data(transfer_k, transfer_data, 3);
}


DataType get_transfer_data(DataType k) {
    return transfer_spline(k);
}


double get_integrand(double k) {
    //double k = x[0];
    double integrand = (4.0 * M_PI) * (1.0 / k) * get_transfer_data(k) * get_transfer_data(k) * get_twopf_data(k) * 1E12;
    return (integrand > 1e-4) ? integrand : 0; 
}

double get_log_integrand(double k, void *user_data) {
    double ell = *(double *)user_data;
    return 2 * get_transfer_data(exp(k)) * get_transfer_data(exp(k)) * get_twopf_data(exp(k)) * 1E12 * ell * (ell + 1);
}

static int Cuba_Integrand(const int *ndim, const double xx[], const int *ncomp, double ff[], void *userdata) {
    #define k xx[0]
    #define dummy xx[1]
    #define f ff[0]

    f = get_integrand(k);
    
    return 0;
}

int Do_Cuba_Integration() {
    #define NCOMP 1
    int comp, nregions, neval, fail;
    double integral[NCOMP], error[NCOMP], prob[NCOMP];

    //printf("\n-------------------- Cuhre test --------------------\n");

    Cuhre(2, 1, Cuba_Integrand, NULL, 1, 1E-14, 1E-14, 0, 0, 100000, 0, NULL, NULL, &nregions, &neval, &fail, integral, error, prob);

    //printf("CUHRE RESULT:\tnregions %d\tneval %d\tfail %d\n", nregions, neval, fail);
    //printf("CUHRE RESULT:\t%.8f +- %.8f\tp = %.3f\n", integral[0], error[0], prob[0]);

    return integral[0];
}

}
