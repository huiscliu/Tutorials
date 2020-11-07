
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    int *v = NULL;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* all gather */
    v = malloc(nprocs * sizeof(*v));

    MPI_Allgather(&rank, 1, MPI_INT, v, 1, MPI_INT, comm);

    free(v);
    MPI_Finalize();

    return 0;
}
