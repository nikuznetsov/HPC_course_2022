#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N_THREADS 4 
#define N 1000000
#define LR 0.1
#define EPOCHS 5000

int main()
{     
    double * x, * y;
    double a, b, a_found = 0, b_found = 0, y_pred, err_tmp;
    double start, end, eps, error = 0, d_da = 0, d_db = 0;

    x = (double *) malloc(N * sizeof(double));
    y = (double *) malloc(N * sizeof(double));

    unsigned int a_seed = time(NULL) + 10;
    unsigned int b_seed = time(NULL) + 5;  
    
    a = -2.0 + ( (double)rand_r(&a_seed) / ((double)RAND_MAX / 4.0) ); 
    b = -3.0 + ( (double)rand_r(&b_seed) / ((double)RAND_MAX / 6.0) );
    
    /*
     *   OPENMP APPROACH
     */

    start = omp_get_wtime();

    #pragma omp parallel num_threads(N_THREADS) \
                         shared(a, b, eps, x, y)
    {
        int tid = omp_get_thread_num();

        unsigned int x_seed = time(NULL) * (tid + 1);
        unsigned int eps_seed = time(NULL) * (tid + 1) + 25;

        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i)
        {   
            x[i] = 0.0 + ( (double)rand_r(&x_seed) / ((double)RAND_MAX / 2.0) );
	    eps = -0.1 + ( (double)rand_r(&eps_seed) / ((double)RAND_MAX / 0.2) );
            y[i] = a * x[i] + b + eps;
        }
    }

    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {  
	#pragma omp parallel num_threads(N_THREADS) \
	                     shared(a_found, b_found, x, y)
	{
	    #pragma omp for reduction(+:d_da,d_db) \
		            private(y_pred) \
		            schedule(static)
            for (int i = 0; i < N; ++i) 
	    {
	        y_pred = a_found * x[i] + b_found;

	        d_da = d_da + x[i] * (y[i] - y_pred);
	        d_db = d_db + (y[i] - y_pred);
	    }
	}

	d_da = (-2.0 / (double)N) * d_da;
	d_db = (-2.0 / (double)N) * d_db;

	a_found = a_found - (double)LR * d_da;
	b_found = b_found - (double)LR * d_db;
    }
    
    /* print data used for fitting
    printf("------------------\n");
    printf("   x    |   y\n");
    printf("------------------\n");
    for (int i = 0; i < n; ++i)
    {
       printf("%.5f | %.5f\n", x[i], y[i]);
    }
    */
    
    #pragma omp parallel num_threads(N_THREADS) \
                         shared(a_found, b_found, x, y)
    {
        #pragma omp for reduction(+:error) \
	                private(err_tmp) \
	                schedule(static)
        for (int i = 0; i < N; ++i)
        {
	    err_tmp = y[i] - (a_found * x[i] + b_found);
            error = error + err_tmp * err_tmp;
        }
    }
    error = error / (double)N; 

    end = omp_get_wtime();

    printf("Real parameters: a = %.5f b = %.5f\n", a, b);
    printf("Found parameters: a = %.5f b = %.5f\n", a_found, b_found);
    printf("Sum of squared residuals = %.5f\n", error);
    
    double omp_time = end - start;
    printf("\nElapsed time (OpenMP) = %.5f seconds\n", omp_time);  
   
    /*
     *  NON-OPENMP APPROACH
     */

    start = omp_get_wtime();

    a_found = 0;
    b_found = 0;
    error = 0;

    unsigned int x_seed = time(NULL);
    unsigned int eps_seed = time(NULL) + 25;
    
    for (int i = 0; i < N; ++i)
    {   
        x[i] = 0.0 + ( (double)rand_r(&x_seed) / ((double)RAND_MAX / 2.0) );
	eps = -0.1 + ( (double)rand_r(&eps_seed) / ((double)RAND_MAX / 0.2) );
        y[i] = a * x[i] + b + eps;
    }

    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {  
	for (int i = 0; i < N; ++i) 
	{
	    y_pred = a_found * x[i] + b_found;

	    d_da = d_da + x[i] * (y[i] - y_pred);
	    d_db = d_db + (y[i] - y_pred);
	}

	d_da = (-2.0 / (double)N) * d_da;
	d_db = (-2.0 / (double)N) * d_db;

	a_found = a_found - (double)LR * d_da;
	b_found = b_found - (double)LR * d_db;
    }
    
    for (int i = 0; i < N; ++i)
    {
	err_tmp = y[i] - (a_found * x[i] + b_found);
        error = error + err_tmp * err_tmp;
    }
    error = error / (double)N; 

    end = omp_get_wtime(); 
    
    double non_omp_time = end - start;
    printf("Elapsed time (non-OpenMP) = %.5f seconds\n", non_omp_time);
   
    printf("Speed up (N_THREADS = %d) = %.5f\n", N_THREADS, non_omp_time / omp_time);   

    free(x);
    free(y);

    return 0;
}
