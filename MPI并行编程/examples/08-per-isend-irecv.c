
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    int *sb = NULL;
    int *rb = NULL;
    MPI_Status status;
    MPI_Request req;

    /* 100 M */
    int size = 1024 * 1024 * 100;
    int i, j, round = 8;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if (nprocs < 2) {
        printf("At least 2 MPIs are required for this example\n");
        goto end;
    }

    /* process 0 sends info to 1 */
    if (rank == 0) {
        sb = malloc(size * sizeof(*sb));

        /* assemble communication pattern */
        MPI_Send_init(sb, size, MPI_INT, 1, 999, comm, &req);

        for (j = 0; j < round; j++) {
            /* send buffer */
            for (i = 0; i < size; i++) sb[i] = j;

            /* real send */
            MPI_Start(&req);

            /* force to complete */
            MPI_Wait(&req, &status);
        }
    }
    else if (rank == 1) {
        rb = malloc(size * sizeof(*rb));

        /* assemble communication pattern */
        MPI_Recv_init(rb, size, MPI_INT, 0, 999, comm, &req);

        for (j = 0; j < round; j++) {
            /* real receive */
            MPI_Start(&req);

            /* force to complete */
            MPI_Wait(&req, &status);

            printf("received: %d %d\n", rb[0], rb[1]);
        }
    }

    free(sb);
    free(rb);

end:
    MPI_Finalize();

    return 0;
}
