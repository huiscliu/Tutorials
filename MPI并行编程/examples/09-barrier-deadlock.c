
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        printf("rank %d is going to sleep\n", rank);
        sleep(10);
        printf("rank %d done sleeping\n", rank);
    }

    if (rank != 0) {
        printf("rank %d is waiting for 0 to wake up\n", rank);
        MPI_Barrier(comm);
    }

    MPI_Finalize();
    return 0;
}

