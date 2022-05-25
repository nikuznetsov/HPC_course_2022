/*
 *	Compile : mpicc strassen.c -o strassen -lm
 *	Run : mpiexec -n <number of processes> ./strassen <matrix size>
 *	
 *	Number of processes : starting from 4, should be power of 2
 *	Matrix size : should be divisible by number of processes
 */

#include <stdio.h>
#include <mpi.h> 
#include <stdlib.h>
#include <math.h>

#define A(i, j) A[(i) * (n) + (j)]
#define B(i, j) B[(i) * (n) + (j)]
#define C(i, j) C[(i) * (n) + (j)]
    
double * Parallel_Multiply(double * , double * , double * , int, int, int, int, int);
double * Form_Matrix(double * , double * , double * , double * , double * , int);

int main(int argc, char * * argv) 
{
    int n = atoi(argv[1]);
    int my_rank, p, rows, columns;
    double *A, *B, *C;    
    double start, end, strassens_time, parallel_multiply_time;

    MPI_Request req[8];
    MPI_Status status;
    MPI_Datatype mysubarray, subarrtype;
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p); 
   
    A = (double *) malloc(n * n * sizeof(double)); 
    B = (double *) malloc(n * n * sizeof(double)); 
    C = (double *) malloc(n * n * sizeof(double)); 

    
    if (my_rank == 0) 
    { 
      for (int i = 0; i < n; i++) 
      {
        for (int j = 0; j < n; j++) 
	{
          A(i, j) = (double)rand() / (double)(RAND_MAX);  
          B(i, j) = (double)rand() / (double)(RAND_MAX);  
          C(i, j) = 0.0; 
        } 
      } 
    }
   
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    start = MPI_Wtime();
    //Parallel_Multiply(A, B, C, 0, my_rank, p, n, n);
    end = MPI_Wtime();
    
    parallel_multiply_time = (double)(end - start);
     
    MPI_Barrier(MPI_COMM_WORLD); 

    start = MPI_Wtime();

    int starts[2] = {0, 0};
    int bigns[2] = {n, n}; 
    int subns[2] = {n / 2, n / 2};

    MPI_Type_create_subarray(2, bigns, subns, starts, MPI_ORDER_C, MPI_DOUBLE, &mysubarray); 
    MPI_Type_create_resized(mysubarray, 0, (n / 2) * (n / 2) * sizeof(double), &subarrtype);
    MPI_Type_commit(&subarrtype);

    rows = n / 2; 
    columns = n / 2;
    
    int no_of_elements = rows * columns;
    double *local_A = (double *) malloc(no_of_elements * sizeof(double));
    double *local_B = (double *) malloc(no_of_elements * sizeof(double)); 
    double *local_C = (double *) malloc(no_of_elements * sizeof(double)); 
 
    if (my_rank == 0) 
    {
      MPI_Isend(A, 1, subarrtype, 0, 0, MPI_COMM_WORLD, &req[0]); 
      MPI_Irecv(local_A, no_of_elements, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req[0]); 
      MPI_Wait(&req[0], &status);
      
      MPI_Isend(B, 1, subarrtype, 0, 0, MPI_COMM_WORLD, &req[0]); 
      MPI_Irecv(local_B, no_of_elements, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req[0]); 
      MPI_Wait(&req[0], &status);
      
      MPI_Isend(C, 1, subarrtype, 0, 0, MPI_COMM_WORLD, &req[0]);
      MPI_Irecv(local_C, no_of_elements, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req[0]);
      MPI_Wait(&req[0], &status);
      
      MPI_Isend(A + n / 2, 1, subarrtype, 1, 1, MPI_COMM_WORLD, &req[1]); 
      MPI_Isend(B + n / 2, 1, subarrtype, 1, 1, MPI_COMM_WORLD, &req[1]);
      MPI_Isend(C + n / 2, 1, subarrtype, 1, 1, MPI_COMM_WORLD, &req[1]);

      MPI_Isend(A + ((n / 2) * (n)), 1, subarrtype, 2, 2, MPI_COMM_WORLD, &req[2]); 
      MPI_Isend(B + ((n / 2) * (n)), 1, subarrtype, 2, 2, MPI_COMM_WORLD, &req[2]); 
      MPI_Isend(C + ((n / 2) * (n)), 1, subarrtype, 2, 2, MPI_COMM_WORLD, &req[2]); 

      MPI_Isend(A + ((n / 2) * n + (n / 2)), 1, subarrtype, 3, 3, MPI_COMM_WORLD, &req[3]);
      MPI_Isend(B + ((n / 2) * n + (n / 2)), 1, subarrtype, 3, 3, MPI_COMM_WORLD, &req[3]); 
      MPI_Isend(C + ((n / 2) * n + (n / 2)), 1, subarrtype, 3, 3, MPI_COMM_WORLD, &req[3]); 
    }

    if (my_rank == 1) 
    {
      MPI_Irecv(local_A, no_of_elements, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &req[1]); 
      MPI_Irecv(local_B, no_of_elements, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &req[1]); 
      MPI_Irecv(local_C, no_of_elements, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &req[1]); 
      MPI_Wait(&req[1], &status);
    }

    if (my_rank == 2) 
    {
      MPI_Irecv(local_A, no_of_elements, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &req[2]);
      MPI_Irecv(local_B, no_of_elements, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &req[2]);
      MPI_Irecv(local_C, no_of_elements, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &req[2]);
      MPI_Wait(&req[2], &status);
    }

    if (my_rank == 3) 
    {
      MPI_Irecv(local_A, no_of_elements, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &req[3]); 
      MPI_Irecv(local_B, no_of_elements, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &req[3]); 
      MPI_Irecv(local_C, no_of_elements, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &req[3]); 
      MPI_Wait(&req[3], &status);
    }

    MPI_Type_free(&subarrtype);
  
    MPI_Barrier(MPI_COMM_WORLD);

    double *T1 = (double *) malloc(sizeof(double) * no_of_elements); 
    double *T2 = (double *) malloc(sizeof(double) * no_of_elements);
    double *T3 = (double *) malloc(sizeof(double) * no_of_elements); 
    double *T4 = (double *) malloc(sizeof(double) * no_of_elements);
    double *T5 = (double *) malloc(sizeof(double) * no_of_elements);
    double *T6 = (double *) malloc(sizeof(double) * no_of_elements); 
    double *T7 = (double *) malloc(sizeof(double) * no_of_elements);
    double *T8 = (double *) malloc(sizeof(double) * no_of_elements);
    double *T9 = (double *) malloc(sizeof(double) * no_of_elements);
    double *T10 = (double *) malloc(sizeof(double) * no_of_elements);
    double *T11 = (double *) malloc(sizeof(double) * no_of_elements);
    double *T12 = (double *) malloc(sizeof(double) * no_of_elements);
    double *T13 = (double *) malloc(sizeof(double) * no_of_elements);
    double *T14 = (double *) malloc(sizeof(double) * no_of_elements);
    double *P1 = (double *) malloc(sizeof(double) * (no_of_elements));
    double *P2 = (double *) malloc(sizeof(double) * (no_of_elements));
    double *P3 = (double *) malloc(sizeof(double) * (no_of_elements));
    double *P4 = (double *) malloc(sizeof(double) * (no_of_elements));
    double *P5 = (double *) malloc(sizeof(double) * (no_of_elements));
    double *P6 = (double *) malloc(sizeof(double) * (no_of_elements));
    double *P7 = (double *) malloc(sizeof(double) * (no_of_elements));
    
    if (my_rank == 0) 
    {
      double *local_A22 = (double *) malloc(sizeof(double) * no_of_elements);
      double *local_B22 = (double *) malloc(sizeof(double) * no_of_elements);

      double *local_A12 = (double *) malloc(sizeof(double) * no_of_elements);
      double *local_B21 = (double *) malloc(sizeof(double) * no_of_elements);

      MPI_Isend(local_A, no_of_elements, MPI_DOUBLE, 1, 100, MPI_COMM_WORLD, &req[0]);
      MPI_Isend(local_A, no_of_elements, MPI_DOUBLE, 3, 500, MPI_COMM_WORLD, &req[4]);
      MPI_Isend(local_B, no_of_elements, MPI_DOUBLE, 2, 400, MPI_COMM_WORLD, &req[3]);
      MPI_Isend(local_B, no_of_elements, MPI_DOUBLE, 3, 600, MPI_COMM_WORLD, &req[4]);

      MPI_Irecv(local_A22, no_of_elements, MPI_DOUBLE, 3, 100, MPI_COMM_WORLD, &req[0]);
      MPI_Irecv(local_B22, no_of_elements, MPI_DOUBLE, 3, 200, MPI_COMM_WORLD, &req[0]);

      MPI_Wait(&req[0], &status);

      MPI_Irecv(local_A12, no_of_elements, MPI_DOUBLE, 1, 100, MPI_COMM_WORLD, &req[1]);
      MPI_Wait(&req[1], &status);

      MPI_Irecv(local_B21, no_of_elements, MPI_DOUBLE, 2, 200, MPI_COMM_WORLD, &req[2]);
      MPI_Wait(&req[0], &status);

      for (int i = 0; i < no_of_elements; i++) 
      {
        *(T1 + i) = *(local_A + i) + *(local_A22 + i);  
        *(T2 + i) = *(local_B + i) + *(local_B22 + i); 
        *(T3 + i) = *(local_A12 + i) - *(local_A22 + i); 
        *(T4 + i) = *(local_B21 + i) + *(local_B22 + i);
      }
      
      local_A22 = NULL;
      local_B22 = NULL;
      local_A12 = NULL;
      local_B21 = NULL;
      
      free(local_A22);
      free(local_B22);
      free(local_A12);
      free(local_B21);
    }

    if (my_rank == 1) 
    {  
      double *local_A11 = (double *) malloc(sizeof(double) * no_of_elements);
      double *local_B22 = (double *) malloc(sizeof(double) * no_of_elements);

      MPI_Isend(local_A, no_of_elements, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &req[1]);
      MPI_Isend(local_B, no_of_elements, MPI_DOUBLE, 3, 500, MPI_COMM_WORLD, &req[1]); 

      MPI_Irecv(local_A11, no_of_elements, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &req[0]); 
      MPI_Irecv(local_B22, no_of_elements, MPI_DOUBLE, 3, 200, MPI_COMM_WORLD, &req[0]); 

      MPI_Wait(&req[0], &status);

      for (int i = 0; i < no_of_elements; i++) 
      { 
        *(T5 + i) = *(local_B + i) - *(local_B22 + i);  
        *(T6 + i) = *(local_A11 + i) + *(local_A + i); 
        *(T7 + i) = *(local_B22 + i); 
        *(T14 + i) = *(local_A11 + i);
      }
      free(local_A11);
      free(local_B22);
    }

    if (my_rank == 2) 
    {
      double *local_A22 = (double *) malloc(sizeof(double) * no_of_elements); 
      double *local_B11 = (double *) malloc(sizeof(double) * no_of_elements);

      MPI_Isend(local_B, no_of_elements, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD, &req[2]); 
      MPI_Isend(local_A, no_of_elements, MPI_DOUBLE, 3, 500, MPI_COMM_WORLD, &req[2]);

      MPI_Irecv(local_A22, no_of_elements, MPI_DOUBLE, 3, 100, MPI_COMM_WORLD, &req[3]);
      MPI_Wait(&req[3], &status);
      
      MPI_Irecv(local_B11, no_of_elements, MPI_DOUBLE, 0, 400, MPI_COMM_WORLD, &req[3]); 
      MPI_Wait(&req[3], &status);

      for (int i = 0; i < no_of_elements; i++) 
      { 
        *(T8 + i) = *(local_A + i) + *(local_A22 + i);  
        *(T9 + i) = *(local_B11 + i); 
        *(T10 + i) = *(local_A22 + i);
        *(T11 + i) = *(local_B + i) - *(local_B11 + i); 
      }
      free(local_A22);
      free(local_B11);
    }
  
    if (my_rank == 3) 
    {
      double *local_A21 = (double *) malloc(sizeof(double) * no_of_elements); 
      double *local_A11 = (double *) malloc(sizeof(double) * no_of_elements);
      double *local_B12 = (double *) malloc(sizeof(double) * no_of_elements); 
      double *local_B11 = (double *) malloc(sizeof(double) * no_of_elements);
     
      MPI_Isend(local_A, no_of_elements, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &req[0]);
      MPI_Isend(local_B, no_of_elements, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD, &req[0]);
      MPI_Isend(local_B, no_of_elements, MPI_DOUBLE, 1, 200, MPI_COMM_WORLD, &req[0]); 
      MPI_Isend(local_A, no_of_elements, MPI_DOUBLE, 2, 100, MPI_COMM_WORLD, &req[3]); 

      MPI_Irecv(local_A21, no_of_elements, MPI_DOUBLE, 2, 500, MPI_COMM_WORLD, &req[4]); 
      MPI_Irecv(local_B11, no_of_elements, MPI_DOUBLE, 0, 600, MPI_COMM_WORLD, &req[4]); 
      MPI_Irecv(local_B12, no_of_elements, MPI_DOUBLE, 1, 500, MPI_COMM_WORLD, &req[4]); 
      MPI_Irecv(local_A11, no_of_elements, MPI_DOUBLE, 0, 500, MPI_COMM_WORLD, &req[5]); 

      MPI_Wait(&req[4], &status);
      MPI_Wait(&req[5], &status);

      for (int i = 0; i < no_of_elements; i++) 
      {
        *(T12 + i) = *(local_A21 + i) - *(local_A11 + i); 
        *(T13 + i) = *(local_B11 + i) + *(local_B12 + i); 
      }
      free(local_B12);
      free(local_A21);
      free(local_A11);
      free(local_B11);
    }
    
    Parallel_Multiply(T1, T2, P1, 0, my_rank, p, rows, columns); 
    Parallel_Multiply(T3, T4, P7, 0, my_rank, p, rows, columns); 
    Parallel_Multiply(T14, T5, P3, 1, my_rank, p, rows, columns);
    Parallel_Multiply(T6, T7, P5, 1, my_rank, p, rows, columns);   
    Parallel_Multiply(T8, T9, P2, 2, my_rank, p, rows, columns);
    Parallel_Multiply(T10, T11, P4, 2, my_rank, p, rows, columns);
    Parallel_Multiply(T12, T13, P6, 3, my_rank, p, rows, columns);

    if (my_rank == 1) 
    {
      MPI_Isend(P3, no_of_elements, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &req[3]);
      MPI_Isend(P5, no_of_elements, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD, &req[3]); 
    }

    if (my_rank == 2) 
    {
      MPI_Isend(P2, no_of_elements, MPI_DOUBLE, 0, 300, MPI_COMM_WORLD, &req[4]); 
      MPI_Isend(P4, no_of_elements, MPI_DOUBLE, 0, 400, MPI_COMM_WORLD, &req[4]); 
    }

    if (my_rank == 3)  MPI_Isend(P6, no_of_elements, MPI_DOUBLE, 0, 500, MPI_COMM_WORLD, &req[4]); 
   
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (my_rank == 0) 
    {
      MPI_Irecv(P3, no_of_elements, MPI_DOUBLE, 1, 100, MPI_COMM_WORLD, &req[3]); 
      MPI_Irecv(P5, no_of_elements, MPI_DOUBLE, 1, 200, MPI_COMM_WORLD, &req[3]); 
      MPI_Irecv(P2, no_of_elements, MPI_DOUBLE, 2, 300, MPI_COMM_WORLD, &req[4]); 
      MPI_Irecv(P4, no_of_elements, MPI_DOUBLE, 2, 400, MPI_COMM_WORLD, &req[4]); 
      MPI_Irecv(P6, no_of_elements, MPI_DOUBLE, 3, 500, MPI_COMM_WORLD, &req[5]); 

      MPI_Wait(&req[3], &status);
      MPI_Wait(&req[4], &status);
      MPI_Wait(&req[5], &status);
      
      double *C11 = (double *) malloc(sizeof(double) * no_of_elements); 
      double *C12 = (double *) malloc(sizeof(double) * no_of_elements);
      double *C21 = (double *) malloc(sizeof(double) * no_of_elements); 
      double *C22 = (double *) malloc(sizeof(double) * no_of_elements);
      
      for (int i = 0; i < no_of_elements; i++) 
      { 
        *(C11 + i) = *(P1 + i) + *(P4 + i) - *(P5 + i) + *(P7 + i); 
        *(C12 + i) = *(P3 + i) + *(P5 + i);
        *(C21 + i) = *(P2 + i) + *(P4 + i);
        *(C22 + i) = *(P1 + i) + *(P3 + i) - *(P2 + i) + *(P6 + i); 
      }
      
      Form_Matrix(C11, C12, C21, C22, C, n);

      end = MPI_Wtime();
      
      strassens_time = (double)(end - start);
      
      free(C11);
      free(C22);
      free(C12);
      free(C21);
    }
     
    MPI_Barrier(MPI_COMM_WORLD);
    
    free(A);
    free(B);
    free(C); 
    free(local_A);
    free(local_B);
    free(local_C);
    free(P1);
    free(T1);
    free(T2);
    free(T3);
    free(T4);
    free(T5);
    free(T6);
    free(T7);
    free(T8);
    free(T9);
    free(T10);
    free(T11);
    free(T12);
    free(T13);
    free(T14);
    free(P2);
    free(P3);
    free(P4);
    free(P5);
    free(P6);
    free(P7);

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (my_rank == 0) 
    {
      printf("Strassens execution time is %f seconds\n", strassens_time);
      //printf("Parallel Matrix multiplication time is %f seconds\n", parallel_multiply_time);
    }
     
    MPI_Finalize();

    return 0;
  }

double *Parallel_Multiply(double * A, double * B, double * C, int root, int my_rank, int no_of_processors, int rows, int columns) 
{
    double from, to;
    double sum;
    double start, end;
    
    if (rows % no_of_processors != 0) 
    {
      if (my_rank == root) printf("Matrix size not divisible by number of processors\n");
      exit(-1);
    }
    
    int no_of_elements = rows * columns;
    double * result = (double * ) malloc(sizeof(double) * no_of_elements);
    
    from = my_rank * rows / no_of_processors; 
    to = (my_rank + 1) * rows / no_of_processors;
   
    MPI_Bcast(B, no_of_elements, MPI_DOUBLE, root, MPI_COMM_WORLD); 

    int elements_per_proc = no_of_elements / no_of_processors;
    double *sub = malloc(sizeof(double) * elements_per_proc);
    
    MPI_Scatter(A, elements_per_proc, MPI_DOUBLE, sub, elements_per_proc, MPI_DOUBLE, root, MPI_COMM_WORLD); 

    int loop = (to - from);
    int ci = 0;
    
    for (int i = from; i < to; i++) 
    {
      for (int j = 0; j < columns; j++) 
      {
        sum = 0;
        for (int k = 0; k < rows; k++) sum += sub[ci * rows + k] * B[k * rows + j]; // Calculate C = A*B
      
        result[ci * rows + j] = sum;
      }
      ci++;
    }
    
    MPI_Gather(result, no_of_elements / no_of_processors, MPI_DOUBLE, C, no_of_elements / no_of_processors, MPI_DOUBLE, root, MPI_COMM_WORLD);
    
    free(result);

    return C;
  }

double *Form_Matrix(double * C11, double * C12, double * C21, double * C22, double * C, int n) {

    int i2 = 0;
    int i3 = -n / 2;
    int i4 = i3;
    int j2, j3;

    for (int i = 0; i < n; i++) 
    {
      j2 = 0;
      j3 = 0;
      for (int j = 0; j < n; j++) 
      {
        if (j < (n / 2) && (i < n / 2)) C[i * n + j] = C11[i * (n / 2) + j]; 
        if (j >= n / 2 && i < n / 2) 
	{   
          C[i * n + j] = C12[i2 * (n / 2) + j2]; 
          j2++;
        }

        if (j < n / 2 && i >= n / 2) C[i * n + j] = C21[i3 * (n / 2) + j]; 

        if (i >= n / 2 && j >= n / 2) 
	{
          C[i * n + j] = C22[i4 * (n / 2) + j3]; //C22
          j3++;
        }
      }

      i2++; 
      i3++; 
      i4++; 
    }

    return C; 
  }
