
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Status status;
    void *sb = NULL;
    void *rb = NULL;
    int position = 0, k, msg_size, buf_size = 2000, dbl_size = 30;
    double dx[30];

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if (nprocs < 2) {
        printf("At least 2 MPIs are required for this example\n");
        goto end;
    }

    sb = malloc(buf_size);
    rb = malloc(buf_size);

    /* process 0 sends info to 1 */
    if (rank == 0) {
        /* pack 1st */
        k = 0;
        MPI_Pack(&k, 1, MPI_INT, sb, buf_size, &position, comm);

        /* pack 2nd round, dbl_size double */
        for (k = 0; k < dbl_size; k++) dx[k] = M_PI * k;

        MPI_Pack(&dx, dbl_size, MPI_DOUBLE, sb, buf_size, &position, comm);

        /* send message size */
        MPI_Send(&position, 1, MPI_INT, 1, 99, comm);

        /* sent message */
        MPI_Send(sb, position, MPI_PACKED, 1, 999, comm);
    }
    else if (rank == 1) {
        /* receive message size */
        MPI_Recv(&msg_size, 1, MPI_INT, 0, 99, comm, &status);

        /* receive message */
        MPI_Recv(rb, msg_size, MPI_PACKED, 0, 999, comm, &status);

        /* unpack 1st */
        MPI_Unpack(rb, msg_size, &position, &k, 1, MPI_INT, comm);
        printf("1st received: %d\n", k);

        /* unpack, 2nd rount, dbl_size double */
        MPI_Unpack(rb, msg_size, &position, dx, dbl_size, MPI_DOUBLE, comm);

        for (k = 0; k < dbl_size; k++) {
            printf("%d-th received double precision number: %.8g\n", k, dx[k]);
        }
    }

    free(sb);
    free(rb);

end:
    MPI_Finalize();

    return 0;
}
