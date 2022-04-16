#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N_THREADS 4
#define N 100
#define LR 0.00001
#define EPOCHS 5000

int main()
{     
    double * x, * y, * y_pred;
    double a, b, a_found = 0, b_found = 0;
    double start, end, eps, error, d_da = 0, d_db = 0;

    x = (double *) malloc(N * sizeof(double));
    y = (double *) malloc(N * sizeof(double));
    
    y_pred = (double *) malloc(N * sizeof(double));

    srand(time(NULL));  
    
    a = -15.0 + ( (double)rand() / ((double)RAND_MAX / 30.0) ); 
    b = -15.0 + ( (double)rand() / ((double)RAND_MAX / 30.0) );

    //omp_set_num_threads(N_THREADS);
    //#pragma omp parallel for shared(x, y, a, b, eps)
    for (int i = 0; i < N; ++i)
    {   
        x[i] = 15.0 + ( (double)rand() / ((double)RAND_MAX / 85.0) );
	eps = -0.1 + ( (double)rand() / ((double)RAND_MAX / 0.2) );
        y[i] = a * x[i] + b + eps;
    }

    start = omp_get_wtime();
    //#pragma omp parallel for
    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        for (int i = 0; i < N; ++i) y_pred[i] = a_found * x[i] + b_found;

	for (int i = 0; i < N; ++i)
	{
	    d_da = d_da + x[i] * (y[i] - y_pred[i]);
	    d_db = d_db + (y[i] - y_pred[i]);
	}

	d_da = (-2.0 / (double)N) * d_da;
	d_db = (-2.0 / (double)N) * d_db;

	a_found = a_found - (double)LR * d_da;
	b_found = b_found - (double)LR * d_db;
    }
    end = omp_get_wtime();

    printf("Real parameters: a = %.5f b = %.5f\n", a, b);
    printf("Found parameters: a = %.5f b = %.5f\n", a_found, b_found);
    printf("Elapsed time = %.5f seconds\n", (double)(end - start));  
    return 0;
}
