
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    int v = 3;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(comm, &rank);

    printf("rank %d, initial v is %d\n", rank, v);

    if (rank == 0) {
        v = 0;
        printf("rank %d changed v to %d\n", rank, v);
    }

    MPI_Bcast(&v, 1, MPI_INT, 0, comm);

    printf("rank %d, v is %d\n", rank, v);

    MPI_Finalize();
    return 0;
}

