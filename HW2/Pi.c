#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N_THREADS 4

int main()
{
    const size_t N = 1000000000;

    int inside_points = 0;
    
    double start, end;

    omp_set_num_threads(N_THREADS);

    start = omp_get_wtime();

    #pragma omp parallel shared(inside_points)
    {
        int tid = omp_get_thread_num();
	
	unsigned int x_seed = time(NULL) * (tid + 1);
	unsigned int y_seed = time(NULL) * (tid + 1) + 25;

	double x, y;

        #pragma omp for reduction(+:inside_points) private(x, y)
	for (int i = 0; i < N; ++i)
	{
	    x = (double)rand_r(&x_seed) / (double)RAND_MAX;
	    y = (double)rand_r(&y_seed) / (double)RAND_MAX;

	    if (x*x + y*y < 1) ++inside_points;
	}        
    }

    end = omp_get_wtime();
    
    printf("OpenMP approach\n");
    printf("pi = %.16f\n", 4 * (double)inside_points / N);
    printf("Time elapsed: %.7f seconds\n", (double)(end - start));
    
    inside_points = 0;

    start = omp_get_wtime();
  
    unsigned int x_seed = time(NULL);
    unsigned int y_seed = time(NULL) + 25;
    double x, y;

    for (int i = 0; i < N; ++i)
    {
        x = (double)rand_r(&x_seed) / (double)RAND_MAX;
        y = (double)rand_r(&y_seed) / (double)RAND_MAX;

        if (x*x + y*y < 1) ++inside_points;
    }

    end = omp_get_wtime();

    printf("\nSimple approach\n");
    printf("pi = %.16f\n", 4 * (double)inside_points / N);
    printf("Time elapsed: %.7f seconds\n", (double)(end - start));

    return 0;
}
