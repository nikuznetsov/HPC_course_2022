#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void zero_init_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0.0;
        }
    }
}

void rand_init_matrix(double ** matrix, size_t N)
{
    srand(time(NULL));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = rand() / RAND_MAX;
        }
    }
}

double ** malloc_matrix(size_t N)
{
    double ** matrix = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; ++i)
    {   
        matrix[i] = (double *)malloc(N * sizeof(double));
    }
    
    return matrix;
}

void free_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; ++i)
    {   
        free(matrix[i]);
    }
    
    free(matrix);
}

int main()
{
    const size_t N = 2500; // size of an array

    double start, end;   
 
    double ** A, ** B, ** C; // matrices

    int i, j ,n;

    printf("Starting:\n");

    A = malloc_matrix(N);
    B = malloc_matrix(N);
    C = malloc_matrix(N);    

    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);
    
    start = omp_get_wtime();

    // OpenMP Matrix Multiplication
    #pragma omp parallel 
    {
        #pragma omp for private(i, j, n) schedule(static)
        for (n = 0; n < N; ++n)
        {
            for (i = 0; i < N; ++i)
            {
                for (j = 0; j < N; ++j)
                {
                    C[i][j] += A[i][n] * B[n][j];
                }
            }
        }
    }
	
    end = omp_get_wtime();

    printf("Time elapsed (OpenMP): %f seconds.\n", (double)(end - start));
    
    zero_init_matrix(C, N);

    start = omp_get_wtime();

    // Simple Matrix Multiplication
    for (n = 0; n < N; ++n)
    {
        for (i = 0; i < N; ++i)
	{
	    for (j = 0; j < N; ++j)
            {
                C[i][j] += A[i][n] * B[n][j];
            }
        }
    }

    end = omp_get_wtime();

    printf("Time elapsed (Without OpenMP): %f seconds.\n", (double)(end - start));

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
