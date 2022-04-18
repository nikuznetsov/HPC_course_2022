#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define N_THREADS 8 
#define TOL 0.000001
#define N 2000
#define MAX_ITER 5000

int main()
{	
    srand(time(NULL));

    double ** A = (double **)calloc(N, sizeof(double*));
    
    for (int i = 0; i < N; ++i)  A[i] = (double *)calloc(N, sizeof(double)); 

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j) 
	{
	    if (i != j) A[i][j] = -9.0 + ( (double)rand() / ((double)RAND_MAX / 18.0) );
	    else A[i][i] = 20.0 + ( (double)rand() / ((double)RAND_MAX / 120.0) );
	}
    }

    double * b = (double *)calloc(N, sizeof(double));
    
    for (int i = 0; i < N; ++i) 
    {
	b[i] = ( (double)rand() / ((double)RAND_MAX / 20.0) );
    }

    double * x = (double *)calloc(N, sizeof(double));

    double * dx = (double *)calloc(N, sizeof(double));

    double start = omp_get_wtime();

    for (int k = 0; k < MAX_ITER; ++k)
    {
        double error = 0.0, sigma;

	#pragma omp parallel for shared(A, b, x, dx) \
	                         private(sigma) \
	                         num_threads(N_THREADS) \
	                         reduction(+:error) \
	                         schedule(static)
	for (int i = 0; i < N; ++i) // loop is not perfectly nested can't use collapse
	{ 
	    dx[i] = b[i];
	    sigma = 0.0;
            
	    for (int j = 0; j < N; ++j) 
	    {
	        sigma = sigma + A[i][j] * x[j];
	    }

	    dx[i] = dx[i] - sigma;
	    dx[i] = dx[i] / A[i][i];
	    x[i] = x[i] + dx[i];
	    error = error + dx[i] * dx[i];
        }

	if (sqrt(error) <= TOL) break;
    }

    double end = omp_get_wtime();
    
    /* Print the system and solution
    printf("System of equations:\n");
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; j++) 
	{
	    if (j == 0) printf("| %.2f  ", A[i][j]);
	    else if (j == N-1) printf("%.2f |  ", A[i][j]);
	    else printf("%.2f  ", A[i][j]);
	}
	printf("| x%d |  =  | %.2f |\n", i+1, b[i]);
    }

    printf("\nSolution:\n");

    for (int i = 0; i < N; ++i)
    {
        printf("x%i = %.3f\n", i+1, x[i]);
    }
    */

    printf("Time elapsed OpenMP (N_THREADS = %d) = %.5f seconds\n", N_THREADS, (double)(end - start));
    
    free(dx);
    free(A);
    free(b);
    free(x);

    return 0;
}
