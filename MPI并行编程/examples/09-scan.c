
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    int i, sum = 0;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* i is rank, sum is sum of all rank IDs */
    i = rank + 1;
    MPI_Scan(&i, &sum, 1, MPI_INT, MPI_SUM, comm);

    printf("correct answer is : %d, computed result is %d\n", (i * (i + 1)) / 2, sum);

    MPI_Finalize();
    return 0;
}
