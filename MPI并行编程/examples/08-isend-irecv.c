
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
    int i;

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
        int flag = 0;

        sb = malloc(size * sizeof(*sb));

        for (i = 0; i < size; i++) sb[i] = size - 1 - i;

        /* send */
        MPI_Isend(sb, size, MPI_INT, 1, 999, comm, &req);

        /* test if Isend completes */
        MPI_Test(&req, &flag, &status);

        if (flag) {
            printf("send completed\n");
        }
        else {
            printf("send not completed yet. now force send to complete\n");

            /* force to complete */
            MPI_Wait(&req, &status);
        }
    }
    else if (rank == 1) {
        int flag = 0;

        rb = malloc(size * sizeof(*rb));

        MPI_Irecv(rb, size, MPI_INT, 0, 999, comm, &req);

        if (flag) {
            printf("receive completed\n");
        }
        else {
            printf("receive not completed yet. now force send to complete\n");

            /* force to complete */
            MPI_Wait(&req, &status);
        }

        printf("received: %d %d\n", rb[0], rb[1]);
    }

    free(sb);
    free(rb);

end:
    MPI_Finalize();

    return 0;
}
