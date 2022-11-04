/*
 * Profile test for fftautocorr, compared with FFTW and naive integral
 *
 *  Copyright (C) 2021 CareF
 *  \author CareF
 *  Licensed under a 3-clause BSD style license - see LICENSE.md
 */

#include "fftautocorr.h"
#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <time.h>
#include <math.h>
#include <limits.h>

#define ERR 1E-8
#define SQ(x) (x)*(x)

int try_fftw(const double *data, int N, clock_t *plan_time, clock_t *exec_time) {
    clock_t t0, t;
    fftw_plan pr2c, pc2r;
    t0 = clock();
    /* Calculate auto-correlation using FFTW */
    double *infftw = fftw_malloc(2 * N * sizeof(double));
    fftw_complex *freq = fftw_malloc((N+1) * sizeof(fftw_complex));
    pr2c = fftw_plan_dft_r2c_1d(2*N, infftw, freq, FFTW_ESTIMATE);
    pc2r = fftw_plan_dft_c2r_1d(2*N, freq, infftw, FFTW_ESTIMATE);
    t = clock();
    *plan_time += t - t0;
    t0 = clock();
    for(int i=0; i<N; i++) {
        infftw[i] = data[i];
    }
    for(int i=N; i<2*N; i++) {
        infftw[i] = 0;
    }
    fftw_execute(pr2c);
    for(int i=0; i<N+1; i++) {
        freq[i][0] = SQ(freq[i][1]) + SQ(freq[i][0]);
        freq[i][1] = 0;
    }
    fftw_execute(pc2r);
    for(int i=0; i<N; i++) {
        infftw[i] /= (2*N);
    }
    t = clock();
    *exec_time += t - t0;

    fftw_destroy_plan(pc2r);
    fftw_destroy_plan(pr2c);
    fftw_free(infftw);
    fftw_free(freq);
    return 0;
}

int try_auto(const double *data, int N, clock_t *plan_time, clock_t *exec_time) {
    clock_t t0, t;
    t0 = clock();
    /* Calculate auto-correlation using fftautocorr */
    autocorr_plan fftauto_plan = make_autocorr_plan(N);
    double *fftauto = malloc(mem_len(fftauto_plan) * sizeof(double));
    t = clock();
    *plan_time += t - t0;
    t0 = clock();
    for(int i=0; i<N; i++) {
        fftauto[i] = data[i];
    }
    for(int i=N; i<mem_len(fftauto_plan); i++) {
        fftauto[i] = 0;
    }
    if(autocorr_p(fftauto_plan, fftauto) != 0) {
        printf("Autocorr executing failed.\n");
        return -1;
    }
    t = clock();
    *exec_time += t - t0;

    destroy_autocorr_plan(fftauto_plan);
    free(fftauto);
    return 0;
}

int testAgainst(int N, int trails) {
    double *data = malloc(N * sizeof(double));
    for(int i=0; i<N; i++) {
        data[i] = (double)rand() / (double)(RAND_MAX);
        /* printf("%f\n", in[i]); */
    }
    clock_t t0, t;
    printf("Testing auto-correlation for length %d:\n", N);

    if (N < 2E6) {
        t0=clock();
        /* Naive algorithm */
        double *nAuto = malloc(N * sizeof(double));
        for(int i = 0; i < N; i++) {
            nAuto[i] = 0;
            for(int j = 0; j < N-i; j++) {
                nAuto[i] += data[j] * data[j+i];
            }
        }
        t = clock();
        printf("Naive autocorr time: %g s\n", (t-t0)/(double)CLOCKS_PER_SEC);
    }
    clock_t fftw_plan_time = 0, fftw_exec_time = 0;
    clock_t auto_plan_time = 0, auto_exec_time = 0;

    printf("Test %d times, remaining: ", trails);
    for (; trails > 0; trails--) {
        printf("%d ", trails);
        if (rand() % 2) {
            try_fftw(data, N, &fftw_plan_time, &fftw_exec_time);
            try_auto(data, N, &auto_plan_time, &auto_exec_time);
        } else {
            try_auto(data, N, &auto_plan_time, &auto_exec_time);
            try_fftw(data, N, &fftw_plan_time, &fftw_exec_time);
        }
    }
    putchar('\n');

    printf("FFTW planning time: %g s\n", fftw_plan_time/(double)CLOCKS_PER_SEC);
    printf("FFTW execution time: %g s\n", fftw_exec_time/(double)CLOCKS_PER_SEC);
    printf("autocorr planning time: %g s\n", auto_plan_time/(double)CLOCKS_PER_SEC);
    printf("autocorr execution time: %g s\n", auto_exec_time/(double)CLOCKS_PER_SEC);

    free(data);
    return 0;
}

int main(int argc, char *argv[]) {
    int trails = 10;
    if (argc > 2) {
        printf("Wrong number of arguments.\n");
        return -1;
    } else if (argc == 2) {
        trails = (int) strtol(argv[1], NULL, 0);
        if (trails <= 0) {
            printf("Illegal number of trails.\n");
            return -1;
        }
    }
    const int Ls[] = {
        1<<15,
        1<<18,
        1<<20,
        4782969, /* = 3^14 */
        9765625,  /* = 5^10 */
        5764801,  /* = 7^8 */
        4561727,  /* prime */
        // INT_MAX/4,
        // 295245/2, /* padded to odd 295245 */
    };
    size_t n = sizeof(Ls)/sizeof(Ls[0]);
    for(int i = 0; i < n; ++i) {
        if(testAgainst(Ls[i], trails) != 0) {
            return -1;
        }
        putchar('\n');
    }
    return 0;
}
