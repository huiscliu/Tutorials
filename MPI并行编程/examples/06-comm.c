
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;

    /* init env */
    MPI_Init(&argc, &argv);

    /* get total number of processes */
    MPI_Comm_size(comm, &nprocs);

    /* get proess ID */
    MPI_Comm_rank(comm, &rank);

    printf("Hello World! This is process: %d / %d\n", rank, nprocs);

    MPI_Finalize();

    return 0;
}
