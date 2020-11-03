
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int i, rank, nprocs;
    int *v = NULL;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* rank 0 gather */
    if (rank == 0) v = malloc(nprocs * sizeof(*v));

    MPI_Gather(&rank, 1, MPI_INT, v, 1, MPI_INT, 0, comm);

    if (rank == 0) {
        for (i = 0; i < nprocs; i++) {
            printf("gathered: %d\n", v[i]);
        }
    }


    free(v);

    MPI_Finalize();
    return 0;
}

