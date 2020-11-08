
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Status status;
    void *sb = NULL;
    void *rb = NULL;
    int position = 0, k, size;
    double pi;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if (nprocs < 2) {
        printf("At least 2 MPIs are required for this example\n");
        goto end;
    }

    sb = malloc(1000);
    rb = malloc(1000);

    /* process 0 sends info to 1 */
    if (rank == 0) {
        /* pack 1st */
        k = 0;
        MPI_Pack(&k, 1, MPI_INT, sb, 1000, &position, comm);

        /* pack 2nd */
        pi = 3.1415926;
        MPI_Pack(&pi, 1, MPI_DOUBLE, sb, 1000, &position, comm);

        /* send message size */
        MPI_Send(&position, 1, MPI_INT, 1, 99, comm);

        /* sent message */
        MPI_Send(sb, position, MPI_PACKED, 1, 999, comm);
    }
    else if (rank == 1) {
        /* receive message size */
        MPI_Recv(&size, 1, MPI_INT, 0, 99, comm, &status);

        /* receive message */
        MPI_Recv(rb, size, MPI_PACKED, 0, 999, comm, &status);

        /* unpack 1st */
        MPI_Unpack(rb, size, &position, &k, 1, MPI_INT, comm);
        printf("1st received: %d\n", k);

        /* unpack 2nd */
        MPI_Unpack(rb, size, &position, &pi, 1, MPI_DOUBLE, comm);
        printf("2nt received: %.8g\n", pi);
    }

    free(sb);
    free(rb);

end:
    MPI_Finalize();

    return 0;
}
