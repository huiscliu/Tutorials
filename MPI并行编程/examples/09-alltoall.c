
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int i, rank, nprocs;
    int *rb = NULL;
    int *sb = NULL;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* buffers */
    rb = malloc(nprocs * sizeof(*rb));
    sb = malloc(nprocs * sizeof(*sb));

    for (i = 0; i < nprocs; i++) sb[i] = rank;

    MPI_Alltoall(sb, 1, MPI_INT, rb, 1, MPI_INT, comm);

    if (rank == 0) {
        for (i = 0; i < nprocs; i++) {
            printf("gathered: %d\n", rb[i]);
        }
    }


    free(rb);
    free(sb);

    MPI_Finalize();
    return 0;
}

