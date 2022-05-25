/*
 *	Compile : gcc matmul_simple_omp.c -o matmul -fopenmp
 *	Run: ./matmul
 *	     You can run like this to collect data into file - ./matmul > out.txt
 */

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

int calculate(int N, int* simple_f, int* omp_f)
{
    //int N = a; // size of an array

    double start, end;   
 
    double ** A, ** B, ** C; // matrices

    int i, j ,n;

    //printf("Starting:\n");

    A = malloc_matrix(N);
    B = malloc_matrix(N);
    C = malloc_matrix(N);    

    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    if (*omp_f == 1)
    {
    
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
 
    printf("N = %d (OpenMP) %f seconds\n", N, (double)(end - start));
    if ( (double)(end-start) > 15.0) *omp_f = 0;
    }
    
    if (*simple_f == 1)
    { 
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

    printf("N = %d (Simple) %f seconds\n", N, (double)(end - start));
    if ( (double)(end-start) > 15.0 ) *simple_f = 0;
    }

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    //return simple_flag, omp_flag;
}

int main()
{   
    int simple_flag = 1, omp_flag = 1;
    int N = 1;

    while ( (simple_flag == 1) || (omp_flag == 1) ) 
    {
	//printf("%d %d\n", simple_flag, omp_flag);
        calculate(N, &simple_flag, &omp_flag);
	N = N + 50;
    }

    return 0;
}
