#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N_THREADS 4
#define N 100

int main()
{     
    int lower = -9, upper = 11;

    double * x, * y;
    double a, b, a_found = 0, b_found = 0;
    double start, end, eps, error;

    x = (double *) malloc(N * sizeof(double));
    y = (double *) malloc(N * sizeof(double));

    srand(time(NULL));  
    
    a = 2 * (rand() % 100) / 50.0 - 1;
    b = 3 * (rand() % 1000) / 500.0 - 2;

    omp_set_num_threads(N_THREADS);
    #pragma omp parallel for shared(x, y, a, b, eps)
    for (int i = 0; i < N; ++i)
    {   
        x[i] = (double)(rand() % 1000) / 1000;
	eps = (double)(2*(rand() % 1000) / 1000 - 1) / 10;
        y[i] = a * x[i] + b + eps;
	//printf("thread = %d i = %d\n", omp_get_thread_num(), i);
    }

    start = omp_get_wtime();
    //#pragma omp parallel for
    end = omp_get_wtime();

    printf("Real parameters: a = %.5f b = %.5f\n", a, b);
    printf("Elapsed time = %.5f seconds\n", (double)(end - start));  
    return 0;
}
