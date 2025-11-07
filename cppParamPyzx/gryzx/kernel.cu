#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <windows.h>

using namespace std;

// TODO: Consider switching from coefficients of root2 to powers of it, or denominators of a 1/x coefficient - so as to make the maths simpler and quicker to compute (i.e. 1+1 is easier than 0.5*0.5 or whatever)
//       This way we could also ensure exact storing of values, and also we could avoid using doubles

const double ROOT2 = 1.414213562373095;
string dir = "";

// INIT CONSTS:

int N_PARAMS;
int N_TERMS;

int MAX_NL_COUNT;
int NL_SUBTERMS;
int NL_ELEMS;

int MAX_PIP_COUNT;
int PIP_SUBTERMS;
int PIP_ELEMS;

int MAX_PP_COUNT;
int PP_SUBTERMS;
int PP_ELEMS;

// INIT PARAM VALS SET:

char* d_paramVals;

// INIT NODE-LIKE TERMS:

char* nl_rowParams, * d_nl_rowParams, * d_nl_rowParamsRO; // (RO = Read-only copy)
char* nl_rowMults, * d_nl_rowMults;
char* nl_rowTypes, * d_nl_rowTypes;
char* nl_rowConsts, * d_nl_rowConsts;
char* d_nl_rowsum;
//double* d_nl_alpha, * d_nl_beta, * d_nl_gamma, * d_nl_delta;
//double* d_fin_terms_alpha, * d_fin_terms_beta, * d_fin_terms_gamma, * d_fin_terms_delta;
double* d_nl_real, * d_nl_imag;
double* d_fin_terms_real, * d_fin_terms_imag;

// PI-PAIR TERMS:

char* pip_rowParams_A, * d_pip_rowParams_A, * d_pip_rowParamsRO_A; // (RO = Read-only copy)
char* pip_rowParams_B, * d_pip_rowParams_B, * d_pip_rowParamsRO_B;
char* pip_rowMults, * d_pip_rowMults;
char* pip_rowPi_A, * d_pip_rowPi_A;
char* pip_rowPi_B, * d_pip_rowPi_B;
char* d_pip_rowsum_A, * d_pip_rowsum_B;
char* d_pip_alpha;

// PHASE-PAIR TERMS:

char* pp_rowParams_A, * d_pp_rowParams_A, * d_pp_rowParamsRO_A; // (RO = Read-only copy)
char* pp_rowParams_B, * d_pp_rowParams_B, * d_pp_rowParamsRO_B;
char* pp_rowMults, * d_pp_rowMults;
char* pp_const_A, * d_pp_const_A;
char* pp_const_B, * d_pp_const_B;
char* d_pp_rowsum_A, * d_pp_rowsum_B;
double* d_pp_real, * d_pp_imag;

// STATIC FACTORS:

char* st_phase, * d_st_phase;
int* st_power2, * d_st_power2;
float* st_ff_re, * d_st_ff_re;
float* st_ff_im, * d_st_ff_im;

//

typedef struct {
    double real, imag;
} scalar;

__global__
void sub_in(int n, int n_subterms, char* datRO, char* dat, char* pVals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n - 1) return;

    dat[i] = datRO[i] * pVals[int(i / n_subterms)]; //  TODO: Could this be more efficient if pVals were in shared mem or something??
}

__global__
void nl_sum_row(int n_subterms, int n_params, char* dat, char* rowsum) { // TODO - this is a TEMP inefficient method
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n_subterms - 1) return;

    rowsum[i] = 0;

    for (int k = 0; k < n_params; ++k) {
        rowsum[i] += dat[i + (k * n_subterms)];
    }

    rowsum[i] = rowsum[i] % 2;
}

__global__
void pip_sum_row(int n_subterms, int n_params, char* dat, char* rowPi, char* rowsum) { // TODO - this is a TEMP inefficient method
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n_subterms - 1) return;

    rowsum[i] = rowPi[i];

    for (int k = 0; k < n_params; ++k) {
        rowsum[i] += dat[i + (k * n_subterms)];
    }

    rowsum[i] = rowsum[i] % 2;
}

__global__
void nl_calc_terms(int n, char* rowMult, char* rowType, char* rowConst, char* rowsum, double* real, double* imag) { // TODO - is there a better way than using a case statement?
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n - 1) return;

    const double root2 = 1.414213562373095;
    // Note: We could actually remove the %8 in the line below if we added "case 8:" immediately after "case 0" and sharing the same block.
    char theta = (4 * rowsum[i] + rowConst[i]) % 8; // theta of term: { 1 + e^[(theta/4)*i*pi] }
    double factor_re = 0.0, factor_im = 0.0;    //  TODO - maybe make some fraction data structure, or otherwise avoid doubles and opt for more reliable int or char? (Might just need to store denominator for some??)

    double ttype = double(rowType[i]) / 4;
    theta = int(double(theta) * ttype);

    switch (theta) {
    case 0: // theta/4 = 0     ==>   term = 1
        factor_re = 1.0;
        break;
    case 1: // theta/4 = 1/4   ==>   term = (root2)/2 + i*(root2)/2
        factor_re = 0.5*root2;
        factor_im = 0.5*root2;
        break;
    case 2: // theta/4 = 1/2   ==>   term = i
        factor_im = 1.0;
        break;
    case 3: // theta/4 = 3/4   ==>   term = -(root2)/2 + i*(root2)/2
        factor_re = -0.5*root2;
        factor_im = 0.5*root2;
        break;
    case 4: // theta/4 = 1     ==>   term = -1
        factor_re = -1.0;
        break;
    case 5: // theta/4 = 5/4   ==>   term = (root2)/2 - i*(root2)/2
        factor_re = -0.5*root2;
        factor_im = -0.5*root2;
        break;
    case 6: // theta/4 = 3/2   ==>   term = i
        factor_im = -1.0;
        break;
    case 7: // theta/4 = 7/4   ==>   term = (root2)/2 - i*(root2)/2
        factor_re = 0.5*root2;
        factor_im = -0.5*root2;
        break;
    }

    if (rowType[i] == 4) factor_re += 1.0; // If its a node then add 1
    //TEMP:
    if (rowMult[i] == 0) {
        factor_re = 1.0;
        factor_im = 0.0;
    }

    // term := alpha + beta*(root2) + gamma*i + delta*i*(root2)
    real[i] = factor_re;
    imag[i] = factor_im;
}

__global__
void pip_calc_terms(int n, char* rowMult, char* rowsumA, char* rowsumB, char* alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n - 1) return;

    char rowsumProd = rowsumA[i] * rowsumB[i]; // rowsumA AND rowsumB = {0,1}

    char expVal = 1;
    if (rowsumProd == 1 && rowMult[i] == 1) expVal = -1;
    alpha[i] = expVal;
}

__global__
void pp_calc_terms(int n, char* rowMult, char* rowConstA, char* rowConstB, char* rowsumA, char* rowsumB, double* real, double* imag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n - 1) return;

    const double root2 = 1.414213562373095;
    char a = (rowConstA[i] + (rowsumA[i] * 4)) % 8;
    char b = (rowConstB[i] + (rowsumB[i] * 4)) % 8;
    char c = (a + b) % 8;

    double factor_re = 1.0, factor_im = 0.0; //  TODO - maybe make some fraction data structure, or otherwise avoid doubles and opt for more reliable int or char? (Might just need to store denominator for some??)

    // 1 + cexp(a) + cexp(b) - cexp(c)...

    switch (a) { // cexp(a)
    case 0: factor_re += 1.0; break;
    case 1: factor_re += 0.5*root2; factor_im += 0.5 * root2; break;
    case 2: factor_im += 1.0; break;
    case 3: factor_re += -0.5*root2; factor_im += 0.5*root2; break;
    case 4: factor_re += -1.0; break;
    case 5: factor_re += -0.5*root2; factor_im += -0.5*root2; break;
    case 6: factor_im += -1.0; break;
    case 7: factor_re += 0.5*root2; factor_im += -0.5*root2; break;
    }

    switch (b) { // cexp(b)
    case 0: factor_re += 1.0; break;
    case 1: factor_re += 0.5*root2; factor_im += 0.5 * root2; break;
    case 2: factor_im += 1.0; break;
    case 3: factor_re += -0.5*root2; factor_im += 0.5*root2; break;
    case 4: factor_re += -1.0; break;
    case 5: factor_re += -0.5*root2; factor_im += -0.5*root2; break;
    case 6: factor_im += -1.0; break;
    case 7: factor_re += 0.5*root2; factor_im += -0.5*root2; break;
    }

    switch (c) { // cexp(c)
    case 0: factor_re -= 1.0; break;
    case 1: factor_re -= 0.5*root2; factor_im -= 0.5 * root2; break;
    case 2: factor_im -= 1.0; break;
    case 3: factor_re -= -0.5*root2; factor_im -= 0.5*root2; break;
    case 4: factor_re -= -1.0; break;
    case 5: factor_re -= -0.5*root2; factor_im -= -0.5*root2; break;
    case 6: factor_im -= -1.0; break;
    case 7: factor_re -= 0.5*root2; factor_im -= -0.5*root2; break;
    }

    //TEMP:
    if (rowMult[i] == 0) {
        factor_re = 1.0;
        factor_im = 0.0;
    }

    // term := alpha + beta*(root2) + gamma*i + delta*i*(root2)
    real[i] = factor_re;
    imag[i] = factor_im;
}

// Reset all fin_terms:
__global__
void init_fin_terms(int n, double* fin_terms_real, double* fin_terms_imag) {
    int term = blockIdx.x * blockDim.x + threadIdx.x;
    if (term > n - 1) return;

    fin_terms_real[term] = 1.0;
    fin_terms_imag[term] = 0.0;
}

// Note: batch_size is the number of sub-terms in each term. e.g. N_PARAMS=50 & batch_size=5 means that every group of 5 is multiplied together, and the resulting 10 terms are then summed
__global__
void nl_mult_subterms(int n, int batch_size, double* real, double* imag, double* fin_terms_real, double* fin_terms_imag) {
    int term = blockIdx.x * blockDim.x + threadIdx.x;
    if (term > n - 1) return;

    //double finAlpha = 1, finBeta = 0, finGamma = 0, finDelta = 0;
    const double root2 = 1.414213562373095;
    double finReal = fin_terms_real[term];
    double finImag = fin_terms_imag[term];
    double realBuffer, imagBuffer;
    int i;

    for (int subterm = 0; subterm < batch_size; ++subterm) { // for each subterm within this term
        i = (term * batch_size) + subterm;
        realBuffer = (finReal * real[i]) - (finImag * imag[i]);
        imagBuffer = (finReal * imag[i]) + (finImag * real[i]);
        finReal = realBuffer;
        finImag = imagBuffer;

        //if (finReal == 0 && finImag == 0) break;
        // TODO - Could (maybe? - if it seems worth it) add a check here to see if the current total is zero - if so, don't bother multiplying by the other subterms for this term
    }
    fin_terms_real[term] = finReal;
    fin_terms_imag[term] = finImag;
}

__global__
void pip_mult_subterms(int n, int batch_size, char* alpha, double* fin_terms_real, double* fin_terms_imag) {
    int term = blockIdx.x * blockDim.x + threadIdx.x;
    if (term > n - 1) return;

    char mult;
    char finMult = 1;
    int i;

    for (int subterm = 0; subterm < batch_size; ++subterm) { // for each subterm within this term
        i = (term * batch_size) + subterm;
        mult = alpha[i];
        finMult *= mult; // TODO - maybe a quicker way might be to count all the '-1's and all the '+1's and from that deduce whether the final mult = 1 or -1 ?
    }

    fin_terms_real[term] *= finMult; // *{-1 or +1}
    fin_terms_imag[term] *= finMult;
}

__global__
void st_apply_factors(int n, char* phase, int* power2, float* ff_re, float* ff_im, double* fin_terms_real, double* fin_terms_imag) {
    int term = blockIdx.x * blockDim.x + threadIdx.x;
    if (term > n - 1) return;

    const double root2 = 1.414213562373095;
    double finReal = fin_terms_real[term], finImag = fin_terms_imag[term];
    double real = 0.0, imag = 0.0; // TODO - maybe make some fraction data structure, or otherwise avoid doubles and opt for more reliable int or char? (Might just need to store denominator for some??)
    double realBuffer, imagBuffer;

    // Phase factor:
    switch (phase[term]) {
    case 0: real = 1.0; break;
    case 1: real = 0.5*root2; imag = 0.5*root2; break;
    case 2: imag = 1.0; break;
    case 3: real = -0.5*root2; imag = 0.5*root2; break;
    case 4: real = -1.0; break;
    case 5: real = -0.5*root2; imag = -0.5*root2; break;
    case 6: imag = -1.0; break;
    case 7: real = 0.5*root2; imag = -0.5*root2; break;
    }

    realBuffer = (finReal * real) - (finImag * imag);
    imagBuffer = (finReal * imag) + (finImag * real);
    finReal = realBuffer;
    finImag = imagBuffer;

    // Power2 factor:
    double power2_factor = pow(2, double(power2[term]) / 2); // Using the fact that: (root2)^x == 2^(x/2)
    finReal *= power2_factor;
    finImag *= power2_factor;

    // Float factor:
    real = double(ff_re[term]);
    imag = double(ff_im[term]);

    realBuffer = (finReal * real) - (finImag * imag);
    imagBuffer = (finReal * imag) + (finImag * real);
    finReal = realBuffer;
    finImag = imagBuffer;

    //--

    fin_terms_real[term] = finReal;
    fin_terms_imag[term] = finImag;
}

__global__
void sum_terms(int n, int n_terms, int split, int gap, double* fin_terms_real, double* fin_terms_imag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n - 1) return;

    int term = i * split;
    if (term + gap > n_terms - 1) return;

    fin_terms_real[term] += fin_terms_real[term + gap];
    fin_terms_imag[term] += fin_terms_imag[term + gap];
}

void CPU_sum_terms(int n_terms, double* fin_terms_alpha, double* fin_terms_beta, double* fin_terms_gamma, double* fin_terms_delta) { // TODO - could this be done in parallel on the gpu somehow?
    double finAlpha = 0, finBeta = 0, finGamma = 0, finDelta = 0;

    for (int i = 0; i < n_terms; ++i) {
        finAlpha += fin_terms_alpha[i];
        finBeta += fin_terms_beta[i];
        finGamma += fin_terms_gamma[i];
        finDelta += fin_terms_delta[i];
    }

    double approxResult_re = finAlpha + (finBeta * ROOT2);
    double approxResult_im = finGamma + (finDelta * ROOT2);
    std::cout << "\n\nFINAL RESULT...";
    std::cout << "\n\n=> " << finAlpha << " + " << finBeta << "*root2 + " << finGamma << "*i " << finDelta << "*root2*i"; // term expression
    std::cout << "\n~= " << approxResult_re << " + " << approxResult_im << "i\n\n";
}

//--

void cpu_sub_in(int i, char* datRO, char* dat, char* pVals) {
    dat[i] = datRO[i] * pVals[int(i % N_PARAMS)];
}

void cpu_sum_row(int i, char* dat, char* rowsum) {
    int j = i * N_PARAMS;

    rowsum[i] = 0;

    for (int k = 0; k < N_PARAMS; ++k) {
        rowsum[i] += dat[j + k];
    }

    rowsum[i] = rowsum[i] % 2;
}

void cpu_calc_terms(int i, char* rowMult, char* rowType, char* rowConst, char* rowsum, double* alpha, double* beta, double* gamma, double* delta) { // TODO - is there a better way than using a case statement?
    char theta = (4 * rowsum[i] + rowConst[i]) % 8; // theta of term: { 1 + e^[(theta/4)*i*pi] }
    double factor_re = 0.0, factor_root2_re = 0.0, factor_im = 0.0, factor_root2_im = 0.0;

    double ttype = double(rowType[i]) / 4;
    theta = int(double(theta) * ttype);

    switch (theta) {
    case 0: // theta/4 = 0     ==>   term = 1
        factor_re = 1.0;
        break;
    case 1: // theta/4 = 1/4   ==>   term = (root2)/2 + i*(root2)/2
        factor_root2_re = 0.5;
        factor_root2_im = 0.5;
        break;
    case 2: // theta/4 = 1/2   ==>   term = i
        factor_im = 1.0;
        break;
    case 3: // theta/4 = 3/4   ==>   term = (root2)/2 + i*(root2)/2
        factor_root2_re = -0.5;
        factor_root2_im = 0.5;
        break;
    case 4: // theta/4 = 1     ==>   term = -1
        factor_re = -1.0;
        break;
    case 5: // theta/4 = 5/4   ==>   term = (root2)/2 - i*(root2)/2
        factor_root2_re = -0.5;
        factor_root2_im = -0.5;
        break;
    case 6: // theta/4 = 3/2   ==>   term = i
        factor_im = -1.0;
        break;
    case 7: // theta/4 = 7/4   ==>   term = (root2)/2 - i*(root2)/2
        factor_root2_re = 0.5;
        factor_root2_im = -0.5;
        break;
    }

    if (rowType[i] == 4) factor_re += 1.0; // If its a node then add 1
    //TEMP:
    if (rowMult[i] == 0) {
        factor_re = 1.0;
        factor_root2_re = 0.0;
        factor_im = 0.0;
        factor_root2_im = 0.0;
    }

    // term := alpha + beta*(root2) + gamma*i + delta*i*(root2)
    alpha[i] = factor_re;
    beta[i] = factor_root2_re;
    gamma[i] = factor_im;
    delta[i] = factor_root2_im;
}

void cpu_mult_subterms(int term, int batch_size, double* alpha, double* beta, double* gamma, double* delta, double* fin_terms_alpha, double* fin_terms_beta, double* fin_terms_gamma, double* fin_terms_delta) {
    double finAlpha = 1, finBeta = 0, finGamma = 0, finDelta = 0;
    double alphaBuffer, betaBuffer, gammaBuffer, deltaBuffer;

    for (int subterm = 0; subterm < batch_size; ++subterm) {
        int i = (term * batch_size) + subterm;
        alphaBuffer = (finAlpha * alpha[i]) + (2 * finBeta * beta[i]) - (finGamma * gamma[i]) - (2 * finDelta * delta[i]);
        betaBuffer = (finAlpha * beta[i]) + (finBeta * alpha[i]) - (finGamma * delta[i]) - (finDelta * gamma[i]);
        gammaBuffer = (finAlpha * gamma[i]) + (2 * finBeta * delta[i]) + (finGamma * alpha[i]) + (2 * finDelta * beta[i]);
        deltaBuffer = (finAlpha * delta[i]) + (finBeta * gamma[i]) + (finGamma * beta[i]) + (finDelta * alpha[i]);
        finAlpha = alphaBuffer;
        finBeta = betaBuffer;
        finGamma = gammaBuffer;
        finDelta = deltaBuffer;
        // TODO - Could (maybe? - if it seems worth it) add a check here to see if the current total is zero - if so, don't bother multiplying by the other subterms for this term
    }
    fin_terms_alpha[term] = finAlpha;
    fin_terms_beta[term] = finBeta;
    fin_terms_gamma[term] = finGamma;
    fin_terms_delta[term] = finDelta;
}

// Initialise working directory:
bool InitWorkingDir() {
    try {
        char buffer[256];
        GetCurrentDirectoryA(256, buffer);
        dir = string(buffer) + '\\';
        dir += "data\\";
    }
    catch (...) {
        std::cout << "\n[ERROR: FAILED TO LOAD WORKING DIRECTORY]\n";
        return false;
    }
    return true;
}

scalar cudaSample(char pvs[]) {
    // PARAM VALUES:
    cudaMemcpy(d_paramVals, pvs, N_PARAMS * sizeof(char), cudaMemcpyHostToDevice);

    //--

    // Initialise:
    init_fin_terms << <(N_TERMS + 255) / 256, 256 >> > (N_TERMS, d_fin_terms_real, d_fin_terms_imag);

    // NODE-LIKE TERMS:
    sub_in << <(NL_ELEMS + 255) / 256, 256 >> > (NL_ELEMS, NL_SUBTERMS, d_nl_rowParamsRO, d_nl_rowParams, d_paramVals);
    nl_sum_row << <(NL_SUBTERMS + 255) / 256, 256 >> > (NL_SUBTERMS, N_PARAMS, d_nl_rowParams, d_nl_rowsum);
    nl_calc_terms << <(NL_SUBTERMS + 255) / 256, 256 >> > (NL_SUBTERMS, d_nl_rowMults, d_nl_rowTypes, d_nl_rowConsts, d_nl_rowsum, d_nl_real, d_nl_imag);
    
    //auto tStart = std::chrono::steady_clock::now(); //TEMP
    nl_mult_subterms << <(N_TERMS + 255) / 256, 256 >> > (N_TERMS, MAX_NL_COUNT, d_nl_real, d_nl_imag, d_fin_terms_real, d_fin_terms_imag);
    //cudaDeviceSynchronize(); //TEMP
    //auto tEnd = std::chrono::steady_clock::now(); //TEMP
    //auto tDiffTot = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count(); //TEMP
    //std::cout << "nl_mult_subterms time: " << tDiffTot << " us\n"; //TEMP

    // PI-PAIR TERMS:
    sub_in << <(PIP_ELEMS + 255) / 256, 256 >> > (PIP_ELEMS, PIP_SUBTERMS, d_pip_rowParamsRO_A, d_pip_rowParams_A, d_paramVals);
    sub_in << <(PIP_ELEMS + 255) / 256, 256 >> > (PIP_ELEMS, PIP_SUBTERMS, d_pip_rowParamsRO_B, d_pip_rowParams_B, d_paramVals);
    pip_sum_row << <(PIP_SUBTERMS + 255) / 256, 256 >> > (PIP_SUBTERMS, N_PARAMS, d_pip_rowParams_A, d_pip_rowPi_A, d_pip_rowsum_A);
    pip_sum_row << <(PIP_SUBTERMS + 255) / 256, 256 >> > (PIP_SUBTERMS, N_PARAMS, d_pip_rowParams_B, d_pip_rowPi_B, d_pip_rowsum_B);
    pip_calc_terms << <(PIP_SUBTERMS + 255) / 256, 256 >> > (PIP_SUBTERMS, d_pip_rowMults, d_pip_rowsum_A, d_pip_rowsum_B, d_pip_alpha);
    pip_mult_subterms << <(N_TERMS + 255) / 256, 256 >> > (N_TERMS, MAX_PIP_COUNT, d_pip_alpha, d_fin_terms_real, d_fin_terms_imag);

    // PHASE-PAIR TERMS:
    sub_in << <(PP_ELEMS + 255) / 256, 256 >> > (PP_ELEMS, PP_SUBTERMS, d_pp_rowParamsRO_A, d_pp_rowParams_A, d_paramVals);
    sub_in << <(PP_ELEMS + 255) / 256, 256 >> > (PP_ELEMS, PP_SUBTERMS, d_pp_rowParamsRO_B, d_pp_rowParams_B, d_paramVals);
    nl_sum_row << <(PP_SUBTERMS + 255) / 256, 256 >> > (PP_SUBTERMS, N_PARAMS, d_pp_rowParams_A, d_pp_rowsum_A);
    nl_sum_row << <(PP_SUBTERMS + 255) / 256, 256 >> > (PP_SUBTERMS, N_PARAMS, d_pp_rowParams_B, d_pp_rowsum_B);
    pp_calc_terms << <(PP_SUBTERMS + 255) / 256, 256 >> > (PP_SUBTERMS, d_pp_rowMults, d_pp_const_A, d_pp_const_B, d_pp_rowsum_A, d_pp_rowsum_B, d_pp_real, d_pp_imag);
    nl_mult_subterms << <(N_TERMS + 255) / 256, 256 >> > (N_TERMS, MAX_PP_COUNT, d_pp_real, d_pp_imag, d_fin_terms_real, d_fin_terms_imag);

    // STATIC FACTORS:
    st_apply_factors << <(N_TERMS + 255) / 256, 256 >> > (N_TERMS, d_st_phase, d_st_power2, d_st_ff_re, d_st_ff_im, d_fin_terms_real, d_fin_terms_imag);

    // SUM TERMS:
    int split = 2;
    int gap = 1; // = split/2;
    int n_chunks;
    while (gap < N_TERMS) {
        n_chunks = ceil(double(N_TERMS) / double(split));
        sum_terms << <(n_chunks + 255) / 256, 256 >> > (n_chunks, N_TERMS, split, gap, d_fin_terms_real, d_fin_terms_imag);
        split *= 2;
        gap *= 2;
    }

    //--

    // COPY BACK TO CPU:

    double* finReal, * finImag;
    finReal = (double*)malloc(sizeof(double));
    finImag = (double*)malloc(sizeof(double));

    cudaMemcpy(finReal, d_fin_terms_real, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(finImag, d_fin_terms_imag, sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(finGamma, d_fin_terms_gamma, sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(finDelta, d_fin_terms_delta, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    scalar result;
    result.real = *finReal;
    result.imag = *finImag;

    return result;
}

void printResult(scalar result) {
    std::cout << "\n\nFINAL RESULT...";
    std::cout << "\n\n=> " << result.real << " + " << result.imag << "*i\n\n"; // Result
}

bool proc(bool showTimes) {
    auto tStart_init = std::chrono::steady_clock::now();

    string strRow, strCell;
    int cellVal;

    // Read meta data (namely metric sizes) from file:

    string path = dir + "data_meta.csv";
    fstream fileMETA(path, ios::in);
    if (!fileMETA.is_open()) {
        std::cout << "\n[ERROR OPENING FILE: " << path << "]\n";
        fileMETA.close();
        return false;
    }

    getline(fileMETA, strRow); N_PARAMS = stoi(strRow);
    getline(fileMETA, strRow); N_TERMS = stoi(strRow);
    getline(fileMETA, strRow); MAX_NL_COUNT = stoi(strRow);
    getline(fileMETA, strRow); MAX_PIP_COUNT = stoi(strRow);
    getline(fileMETA, strRow); MAX_PP_COUNT = stoi(strRow);

    fileMETA.close();

    NL_SUBTERMS = MAX_NL_COUNT * N_TERMS;
    NL_ELEMS = NL_SUBTERMS * N_PARAMS;

    PIP_SUBTERMS = MAX_PIP_COUNT * N_TERMS;
    PIP_ELEMS = PIP_SUBTERMS * N_PARAMS;

    PP_SUBTERMS = MAX_PP_COUNT * N_TERMS;
    PP_ELEMS = PP_SUBTERMS * N_PARAMS;

    //--------------------------------------------------

    // NODE-LIKE TERMS:

    nl_rowParams = (char*)malloc(NL_ELEMS * sizeof(char));
    nl_rowMults = (char*)malloc(NL_SUBTERMS * sizeof(char));
    nl_rowTypes = (char*)malloc(NL_SUBTERMS * sizeof(char));
    nl_rowConsts = (char*)malloc(NL_SUBTERMS * sizeof(char));

    // Read data from file:

    path = dir + "data_nodes.csv";
    fstream fileNL(path, ios::in);
    if (!fileNL.is_open()) {
        std::cout << "\n[ERROR OPENING FILE: " << path << "]\n";
        fileNL.close();
        return false;
    }

    //getline(file, strRow); // ignore header row
    for (int y = 0; y < NL_SUBTERMS; ++y) { // for each line
        getline(fileNL, strRow);
        stringstream str(strRow);
        getline(str, strCell, ','); nl_rowMults[y] = stoi(strCell);
        getline(str, strCell, ','); nl_rowTypes[y] = stoi(strCell);
        getline(str, strCell, ','); nl_rowConsts[y] = stoi(strCell);

        for (int x = 0; x < N_PARAMS; ++x) { // for each column
            getline(str, strCell, ','); cellVal = stoi(strCell);
            //nl_rowParams[(y * N_PARAMS) + x] = cellVal; // row-major
            nl_rowParams[(x * NL_SUBTERMS) + y] = cellVal; // col-major
        }
    }
    fileNL.close();

    //--------------------------------------------------

    // PI-PAIR TERMS:

    pip_rowParams_A = (char*)malloc(PIP_ELEMS * sizeof(char));
    pip_rowParams_B = (char*)malloc(PIP_ELEMS * sizeof(char));
    pip_rowMults = (char*)malloc(PIP_SUBTERMS * sizeof(char));
    pip_rowPi_A = (char*)malloc(PIP_SUBTERMS * sizeof(char));
    pip_rowPi_B = (char*)malloc(PIP_SUBTERMS * sizeof(char));

    //string strRow, strCell;
    //int cellVal;
    //int x = 0, y = 0;

    // Read data from file:

    path = dir + "data_pipairs.csv";
    fstream filePIP(path, ios::in);
    if (!filePIP.is_open()) {
        std::cout << "\n[ERROR OPENING FILE: " << path << "]\n";
        filePIP.close();
        return false;
    }

    for (int y = 0; y < PIP_SUBTERMS; ++y) { // for each line
        getline(filePIP, strRow);
        stringstream str(strRow);

        getline(str, strCell, ','); pip_rowMults[y] = stoi(strCell);
        getline(str, strCell, ','); pip_rowPi_A[y] = stoi(strCell);
        for (int x = 0; x < N_PARAMS; ++x) { // for each column
            getline(str, strCell, ','); cellVal = stoi(strCell);
            pip_rowParams_A[(x * PIP_SUBTERMS) + y] = cellVal; // col-major
        }

        getline(str, strCell, ','); pip_rowPi_B[y] = stoi(strCell);
        for (int x = 0; x < N_PARAMS; ++x) { // for each column
            getline(str, strCell, ','); cellVal = stoi(strCell);
            pip_rowParams_B[(x * PIP_SUBTERMS) + y] = cellVal; // col-major
        }
    }
    filePIP.close();

    //--------------------------------------------------

    // PHASE-PAIR TERMS:

    pp_rowParams_A = (char*)malloc(PP_ELEMS * sizeof(char));
    pp_rowParams_B = (char*)malloc(PP_ELEMS * sizeof(char));
    pp_rowMults = (char*)malloc(PP_SUBTERMS * sizeof(char));
    pp_const_A = (char*)malloc(PP_SUBTERMS * sizeof(char));
    pp_const_B = (char*)malloc(PP_SUBTERMS * sizeof(char));

    // Read data from file:

    path = dir + "data_phasepairs.csv";
    fstream filePP(path, ios::in);
    if (!filePP.is_open()) {
        std::cout << "\n[ERROR OPENING FILE: " << path << "]\n";
        filePP.close();
        return false;
    }

    for (int y = 0; y < PP_SUBTERMS; ++y) { // for each line
        getline(filePP, strRow);
        stringstream str(strRow);

        getline(str, strCell, ','); pp_rowMults[y] = stoi(strCell);
        getline(str, strCell, ','); pp_const_A[y] = stoi(strCell);
        getline(str, strCell, ','); pp_const_B[y] = stoi(strCell);

        for (int x = 0; x < N_PARAMS; ++x) { // for each column
            getline(str, strCell, ','); cellVal = stoi(strCell);
            pp_rowParams_A[(x * PP_SUBTERMS) + y] = cellVal; // col-major
        }

        for (int x = 0; x < N_PARAMS; ++x) { // for each column
            getline(str, strCell, ','); cellVal = stoi(strCell);
            pp_rowParams_B[(x * PP_SUBTERMS) + y] = cellVal; // col-major
        }
    }
    filePP.close();

    //--------------------------------------------------

    // STATIC FACTORS:

    st_phase = (char*)malloc(N_TERMS * sizeof(char));
    st_power2 = (int*)malloc(N_TERMS * sizeof(int));
    st_ff_re = (float*)malloc(N_TERMS * sizeof(float));
    st_ff_im = (float*)malloc(N_TERMS * sizeof(float));

    // Read data from file:

    path = dir + "data_static.csv";
    fstream fileST(path, ios::in);
    if (!fileST.is_open()) {
        std::cout << "\n[ERROR OPENING FILE: " << path << "]\n";
        fileST.close();
        return false;
    }

    for (int y = 0; y < N_TERMS; ++y) { // for each line
        getline(fileST, strRow);
        stringstream str(strRow);
        getline(str, strCell, ',');  st_phase[y] = stoi(strCell);
        getline(str, strCell, ','); st_power2[y] = stoi(strCell);
        getline(str, strCell, ',');  st_ff_re[y] = stof(strCell);
        getline(str, strCell, ',');  st_ff_im[y] = stof(strCell);
    }
    fileST.close();

    //cudaDeviceSynchronize();

    //--------------------------------------------------
    
    // CUDA MALLOC:

    auto tStart_cudamalloc = std::chrono::steady_clock::now();

    cudaMalloc(&d_paramVals, N_PARAMS * sizeof(char));

    cudaMalloc(&d_nl_rowParams, NL_ELEMS * sizeof(char));
    cudaMalloc(&d_nl_rowParamsRO, NL_ELEMS * sizeof(char));
    cudaMalloc(&d_nl_rowMults, NL_SUBTERMS * sizeof(char));
    cudaMalloc(&d_nl_rowTypes, NL_SUBTERMS * sizeof(char));
    cudaMalloc(&d_nl_rowConsts, NL_SUBTERMS * sizeof(char));
    cudaMalloc(&d_nl_rowsum, NL_SUBTERMS * sizeof(char));
    cudaMalloc(&d_nl_real, NL_SUBTERMS * sizeof(double));
    cudaMalloc(&d_nl_imag, NL_SUBTERMS * sizeof(double));
    cudaMalloc(&d_fin_terms_real, N_TERMS * sizeof(double));
    cudaMalloc(&d_fin_terms_imag, N_TERMS * sizeof(double));

    cudaMalloc(&d_pip_rowParams_A, PIP_ELEMS * sizeof(char));
    cudaMalloc(&d_pip_rowParams_B, PIP_ELEMS * sizeof(char));
    cudaMalloc(&d_pip_rowParamsRO_A, PIP_ELEMS * sizeof(char));
    cudaMalloc(&d_pip_rowParamsRO_B, PIP_ELEMS * sizeof(char));
    cudaMalloc(&d_pip_rowMults, PIP_SUBTERMS * sizeof(char));
    cudaMalloc(&d_pip_rowPi_A, PIP_SUBTERMS * sizeof(char));
    cudaMalloc(&d_pip_rowPi_B, PIP_SUBTERMS * sizeof(char));
    cudaMalloc(&d_pip_rowsum_A, PIP_SUBTERMS * sizeof(char));
    cudaMalloc(&d_pip_rowsum_B, PIP_SUBTERMS * sizeof(char));
    cudaMalloc(&d_pip_alpha, PIP_SUBTERMS * sizeof(char));

    cudaMalloc(&d_pp_rowParams_A, PP_ELEMS * sizeof(char));
    cudaMalloc(&d_pp_rowParams_B, PP_ELEMS * sizeof(char));
    cudaMalloc(&d_pp_rowParamsRO_A, PP_ELEMS * sizeof(char));
    cudaMalloc(&d_pp_rowParamsRO_B, PP_ELEMS * sizeof(char));
    cudaMalloc(&d_pp_rowMults, PP_SUBTERMS * sizeof(char));
    cudaMalloc(&d_pp_const_A, PP_SUBTERMS * sizeof(char));
    cudaMalloc(&d_pp_const_B, PP_SUBTERMS * sizeof(char));
    cudaMalloc(&d_pp_rowsum_A, PP_SUBTERMS * sizeof(char));
    cudaMalloc(&d_pp_rowsum_B, PP_SUBTERMS * sizeof(char));
    cudaMalloc(&d_pp_real, PP_SUBTERMS * sizeof(double));
    cudaMalloc(&d_pp_imag, PP_SUBTERMS * sizeof(double));

    cudaMalloc(&d_st_phase, N_TERMS * sizeof(char));
    cudaMalloc(&d_st_power2, N_TERMS * sizeof(int));
    cudaMalloc(&d_st_ff_re, N_TERMS * sizeof(float));
    cudaMalloc(&d_st_ff_im, N_TERMS * sizeof(float));

    cudaDeviceSynchronize();
    
    //--------------------------------------------------

    // LOAD DATA TO GPU:

    auto tStart_memcpy = std::chrono::steady_clock::now();

    cudaMemcpy(d_nl_rowParamsRO, nl_rowParams, NL_ELEMS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nl_rowMults, nl_rowMults, NL_SUBTERMS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nl_rowTypes, nl_rowTypes, NL_SUBTERMS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nl_rowConsts, nl_rowConsts, NL_SUBTERMS * sizeof(char), cudaMemcpyHostToDevice);

    cudaMemcpy(d_pip_rowParamsRO_A, pip_rowParams_A, PIP_ELEMS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pip_rowParamsRO_B, pip_rowParams_B, PIP_ELEMS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pip_rowMults, pip_rowMults, PIP_SUBTERMS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pip_rowPi_A, pip_rowPi_A, PIP_SUBTERMS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pip_rowPi_B, pip_rowPi_B, PIP_SUBTERMS * sizeof(char), cudaMemcpyHostToDevice);

    cudaMemcpy(d_pp_rowParamsRO_A, pp_rowParams_A, PP_ELEMS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pp_rowParamsRO_B, pp_rowParams_B, PP_ELEMS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pp_rowMults, pp_rowMults, PP_SUBTERMS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pp_const_A, pp_const_A, PP_SUBTERMS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pp_const_B, pp_const_B, PP_SUBTERMS * sizeof(char), cudaMemcpyHostToDevice);

    cudaMemcpy(d_st_phase, st_phase, N_TERMS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_st_power2, st_power2, N_TERMS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_st_ff_re, st_ff_re, N_TERMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_st_ff_im, st_ff_im, N_TERMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    //--------------------------------------------------

    // CUDA KERNEL CALLS:

    auto tEnd_init = std::chrono::steady_clock::now();
    auto tStartCuda = std::chrono::steady_clock::now();

    // One sample:
    char pvs[] = { 1,0,1,0,0 };//,0,0,0,0,0 }; // TODO: init to length N_PARAMS
    scalar result = cudaSample(pvs);

    auto tEndCuda = std::chrono::steady_clock::now();
    printResult(result);

    // 100 random samples:
    auto tStartSample = std::chrono::steady_clock::now();
    auto tEndSample = std::chrono::steady_clock::now();
    auto tDiffSample = std::chrono::duration_cast<std::chrono::microseconds>(tEndSample - tStartSample).count();
    int times[100];
    for (int repeat = 0; repeat < 100; ++repeat) {
        for (int i = 0; i < N_PARAMS; ++i) pvs[i] = rand() % 2;
        tStartSample = std::chrono::steady_clock::now();
        cudaSample(pvs);
        tEndSample = std::chrono::steady_clock::now();
        tDiffSample = std::chrono::duration_cast<std::chrono::microseconds>(tEndSample - tStartSample).count();
        times[repeat] = tDiffSample;
        //std::cout << "\n" << tDiffSample;
    }

    // Get ave, max, and min times:
    int min = 999999999;
    int max = -1;
    int ave = 0;
    for (int i = 0; i < 100; ++i) {
        if (times[i] < min) min = times[i];
        if (times[i] > max) max = times[i];
        ave += times[i];
    }
    ave = int(ave / 100);
    std::cout << "Sample times...\n";
    std::cout << "ave: " << ave << "\n";
    std::cout << "min: " << min << "\n";
    std::cout << "max: " << max << "\n\n";

    //--------------------------------------------------

    auto tDiff_init = std::chrono::duration_cast<std::chrono::microseconds>(tStart_cudamalloc - tStart_init).count();
    auto tDiff_cudamalloc = std::chrono::duration_cast<std::chrono::microseconds>(tStart_memcpy - tStart_cudamalloc).count();
    auto tDiff_memcpy = std::chrono::duration_cast<std::chrono::microseconds>(tEnd_init - tStart_memcpy).count();
    auto tDiff_totsetup = std::chrono::duration_cast<std::chrono::microseconds>(tEnd_init - tStart_init).count();
    auto tDiff_cuda = std::chrono::duration_cast<std::chrono::microseconds>(tEndCuda - tStartCuda).count();
    auto tDiff_tot = std::chrono::duration_cast<std::chrono::microseconds>(tEndCuda - tStart_init).count();
    if (showTimes) {
        std::cout << "Setup times...\n\n";
        std::cout << "Read/malloc time: \t" << tDiff_init << " us\n";
        std::cout << "Cudamalloc  time: \t" << tDiff_cudamalloc << " us\n";
        std::cout << "Memcpy      time: \t" << tDiff_memcpy << " us\n";
        std::cout << "Total setup time: \t" << tDiff_totsetup << " us\n\n";
        std::cout << "Cuda  time: \t" << tDiff_cuda << " us\n";
        std::cout << "total time: \t" << tDiff_tot << " us\n";
    }

    //cudaFree(d_x);
    //cudaFree(d_y);
    //free(x);
    //free(y);

    return true;
}

// TEMP (currently huge amount of overlap between this func and proc()):
bool proc_cpu(bool showTimes) {
    auto tStart = std::chrono::steady_clock::now();

    string path = dir + "data_nodes.csv";
    fstream file(path, ios::in);
    if (!file.is_open()) {
        std::cout << "\n[ERROR OPENING FILE: " << path << "]\n";
        file.close();
        return false;
    }

    //

    const int MAX_NODE_COUNT = 27;
    const int N_TERMS = 68;
    const int N_SUBTERMS = MAX_NODE_COUNT * N_TERMS;
    const int N_ELEMS = N_SUBTERMS * N_PARAMS;

    // PARAM VALUES:
    //             a b c d e f g h i
    char pvs[] = { 1,0,1,1,0,0,0,1,0 }; // <-- PARAM VALUES

    char* paramVals;
    paramVals = (char*)malloc(N_PARAMS * sizeof(char));
    for (int i = 0; i < N_PARAMS; ++i) paramVals[i] = pvs[i];

    // NODE-TYPE TERMS:

    char* rowParams, * rowParamsRO; // (RO = Read-only copy)
    char* rowMults;
    char* rowTypes;
    char* rowConsts;
    char* rowsum;
    double* alpha, * beta, * gamma, * delta;
    double* fin_terms_alpha, * fin_terms_beta, * fin_terms_gamma, * fin_terms_delta;

    rowParams = (char*)malloc(N_ELEMS * sizeof(char));
    rowParamsRO = (char*)malloc(N_ELEMS * sizeof(char));
    rowMults = (char*)malloc(N_SUBTERMS * sizeof(char));
    rowTypes = (char*)malloc(N_SUBTERMS * sizeof(char));
    rowConsts = (char*)malloc(N_SUBTERMS * sizeof(char));
    rowsum = (char*)malloc(N_SUBTERMS * sizeof(char));
    alpha = (double*)malloc(N_SUBTERMS * sizeof(double));
    beta = (double*)malloc(N_SUBTERMS * sizeof(double));
    gamma = (double*)malloc(N_SUBTERMS * sizeof(double));
    delta = (double*)malloc(N_SUBTERMS * sizeof(double));
    fin_terms_alpha = (double*)malloc(N_TERMS * sizeof(double));
    fin_terms_beta = (double*)malloc(N_TERMS * sizeof(double));
    fin_terms_gamma = (double*)malloc(N_TERMS * sizeof(double));
    fin_terms_delta = (double*)malloc(N_TERMS * sizeof(double));

    string strRow, strCell;
    int cellVal;
    int x = 0, y = 0;

    //getline(file, strRow); // ignore header row
    while (getline(file, strRow)) {
        stringstream str(strRow);
        getline(str, strCell, ','); rowMults[y] = stoi(strCell);
        getline(str, strCell, ','); rowTypes[y] = stoi(strCell);
        getline(str, strCell, ','); rowConsts[y] = stoi(strCell);

        x = 0;
        while (getline(str, strCell, ',')) {
            cellVal = stoi(strCell);
            rowParams[(y * N_PARAMS) + x] = cellVal; // row-major
            rowParamsRO[(y * N_PARAMS) + x] = cellVal; // row-major
            //rowParams[(x * N_SUBTERMS) + y] = cellVal; // col-major
            //rowParamsRO[(x * N_SUBTERMS) + y] = cellVal; // col-major
            ++x;
        }
        ++y;
    }

    // PI-PAIR TERMS:

    //TODO

    //--------------------------------------------------

    // FUNCTION CALLS:
    auto tStartProc = std::chrono::steady_clock::now();
    //sub_in << <(N_ELEMS + 255) / 256, 256 >> > (N_ELEMS, N_SUBTERMS, d_rowParamsRO, d_rowParams, d_paramVals);
    //sum_row << <(N_SUBTERMS + 255) / 256, 256 >> > (N_SUBTERMS, d_rowParams, d_rowsum);
    //calc_terms << <(N_SUBTERMS + 255) / 256, 256 >> > (N_SUBTERMS, d_rowMults, d_rowTypes, d_rowConsts, d_rowsum, d_alpha, d_beta, d_gamma, d_delta);
    //mult_subterms << <(N_TERMS + 255) / 256, 256 >> > (N_TERMS, MAX_NODE_COUNT, d_alpha, d_beta, d_gamma, d_delta, d_fin_terms_alpha, d_fin_terms_beta, d_fin_terms_gamma, d_fin_terms_delta);
    for (int i = 0; i < N_ELEMS; ++i) cpu_sub_in(i, rowParamsRO, rowParams, paramVals);
    for (int i = 0; i < N_SUBTERMS; ++i) cpu_sum_row(i, rowParams, rowsum);
    for (int i = 0; i < N_SUBTERMS; ++i) cpu_calc_terms(i, rowMults, rowTypes, rowConsts, rowsum, alpha, beta, gamma, delta);
    for (int i = 0; i < N_TERMS; ++i) cpu_mult_subterms(i, MAX_NODE_COUNT, alpha, beta, gamma, delta, fin_terms_alpha, fin_terms_beta, fin_terms_gamma, fin_terms_delta);
    auto tEndProc = std::chrono::steady_clock::now();
    auto tDiffProc = std::chrono::duration_cast<std::chrono::microseconds>(tEndProc - tStartProc).count();

    //CPU_sum_terms(N_TERMS, fin_terms_alpha, fin_terms_beta, fin_terms_gamma, fin_terms_delta);

    //
    auto tEnd = std::chrono::steady_clock::now();
    auto tDiffTot = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();
    auto tDiffProcFull = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStartProc).count();
    if (showTimes) {
        std::cout << "ProcFull t: \t" << tDiffProcFull << " us\n";
        std::cout << "Proc time:  \t" << tDiffProc << " us\n";
        std::cout << "total time: \t" << tDiffTot << " us\n";
    }
    return true;
}

int main(void) {
    InitWorkingDir();
    proc(true);
    //proc_cpu(true);
}