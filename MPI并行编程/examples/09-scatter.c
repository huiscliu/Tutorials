
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int i, rank, nprocs;
    int *v = NULL, b;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* rank 0 scatter */
    if (rank == 0) {
        v = malloc(nprocs * sizeof(*v));

        for (i = 0; i < nprocs; i++) {
            v[i] = rand();
            printf("buf in rank 0: %d\n", v[i]);
        }
    }

    MPI_Scatter(v, 1, MPI_INT, &b, 1, MPI_INT, 0, comm);

    printf("received by rank %d: %d\n", rank, b);

    free(v);

    MPI_Finalize();
    return 0;
}

