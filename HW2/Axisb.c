#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define N_THREADS 4
#define TOL 0.0001
#define N 3
#define MAX_ITER 1000

int main()
{
    srand(time(NULL));

    double ** A = (double **)calloc(N, sizeof(double*));
    for (int i = 0; i < N; ++i)  A[i] = (double *)calloc(N, sizeof(double));  
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j) A[i][j] = 1 + rand() % 10;
	A[i][i] = N * 10 + rand() % 5;
    }

    double * b = (double *)calloc(N, sizeof(double));
    for (int i = 0; i < N; ++i) b[i] = rand() % 20;

    double * x = (double *)calloc(N, sizeof(double));

    double * dx = (double *)calloc(N, sizeof(double));

    double delxi, numit = 0;

    for (int k = 0; k < MAX_ITER; ++k)
    {
        double norm_error = 0.0;
	for (int i = 0; i < N; ++i)
	{
	    dx[i] = b[i];
	    delxi = 0.0;

	    for (int j = 0; j < N; ++j)
	    {
	        delxi = delxi + A[i][j] * x[j];
	    }

	    dx[i] = dx[i] - delxi;
	    dx[i] = dx[i] / A[i][i];
	    x[i] = x[i] + dx[i];
	    norm_error = norm_error + dx[i] * dx[i];
	}
	if (sqrt(norm_error) <= TOL) break;

        numit = k+1;
    }

    free(dx);

    for (int i = 0; i < N; ++i)
    {
        printf("x%i=%.3f\n", i+1, x[i]);
    }

    return 0;
}
