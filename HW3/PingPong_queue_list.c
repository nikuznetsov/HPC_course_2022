#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

#define LIMIT 500000

int main()
{
    int n = 0;
    char str[LIMIT] = "";

    double start, end;

    srand(time(NULL));

    int flag = 0;
    int send, address;

    MPI_Status status;
    MPI_Init(NULL, NULL);
    
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    send = 0;
    address = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    while (n < LIMIT)
    {
	if (rank == send)
	{
  	    unsigned int seed = time(NULL) * (rank + 1);

	    if (send == address) while (send == address) address = rand_r(&seed) % size;

	    ++n;

	    char tmp_str[10];
	    sprintf(tmp_str, "%d", rank);

	    strncat(str, tmp_str, 5);
	    //printf("%s\n", str);

	    MPI_Ssend(&str, strlen(str)+1, MPI_BYTE, address, 0, MPI_COMM_WORLD);
	    MPI_Ssend(&n, 1, MPI_INT, address, 1, MPI_COMM_WORLD);
	    
	    send = address;
	}
	else // if (rank == address)
	{
	    while(!flag) MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);

	    //printf("RECV %d >>> %d\n", status.MPI_SOURCE, rank);

	    MPI_Recv(&str, strlen(str)+2, MPI_BYTE, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
	    MPI_Recv(&n, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &status);

	    flag = 0;
	    send = rank;
	}
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    MPI_Finalize();

    //printf("\nProc %d DONE\n", rank);

    if (rank == 0) 
    {
        printf("Total time = %f sec\n", (double)(end-start));
	printf("# of itterations = %d\n", LIMIT);
	//printf("Size of message = %ld\n", strlen(str));
    }
    return 0;
}
