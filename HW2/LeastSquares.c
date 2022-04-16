#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N_THREADS 4
#define N 100
#define LR 0.1
#define EPOCHS 5000

int main()
{     
    double * x, * y;
    double a, b, a_found = 0, b_found = 0, y_pred, err_tmp;
    double start, end, eps, error = 0, d_da = 0, d_db = 0;

    x = (double *) malloc(N * sizeof(double));
    y = (double *) malloc(N * sizeof(double));
    
    //y_pred = (double *) malloc(N * sizeof(double));

    srand(time(NULL));  
    
    a = -2.0 + ( (double)rand() / ((double)RAND_MAX / 4.0) ); 
    b = -3.0 + ( (double)rand() / ((double)RAND_MAX / 6.0) );

    //omp_set_num_threads(N_THREADS);
    //#pragma omp parallel for shared(x, y, a, b, eps)
    for (int i = 0; i < N; ++i)
    {   
        x[i] = 0.0 + ( (double)rand() / ((double)RAND_MAX / 2.0) );
	eps = -0.1 + ( (double)rand() / ((double)RAND_MAX / 0.2) );
	//printf("%f\n", eps);
        y[i] = a * x[i] + b + eps;
    }

    start = omp_get_wtime();
    //#pragma omp parallel for
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
    end = omp_get_wtime();
    
    printf("-----------------\n");
    printf("   X    |   Y\n");
    printf("-----------------\n");
    for (int i = 0; i < N; ++i)
    {
       printf("%.5f | %.5f\n", x[i], y[i]);
    }

    for (int i = 0; i < N; ++i)
    {
	err_tmp = y[i] - (a_found * x[i] + b_found);
        error = error + err_tmp * err_tmp;
    }
    error = error / (double)N; 

    printf("\nReal parameters: a = %.5f b = %.5f\n", a, b);
    printf("Found parameters: a = %.5f b = %.5f\n", a_found, b_found);
    printf("Sum of squared residuals = %.5f\n", error);
    printf("\nElapsed time = %.5f seconds\n", (double)(end - start));  
    return 0;
}
