
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    printf("Hello World! This is process: %d / %d\n", rank, nprocs);

    MPI_Finalize();

    return 0;
}
