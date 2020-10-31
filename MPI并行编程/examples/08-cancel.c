
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define round 8

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    int **sb = NULL;
    int **rb = NULL;
    MPI_Status status[round];
    MPI_Request req[round];

    /* 100 M */
    int size = 1024 * 1024 * 100;
    int i, j;

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

        sb = malloc(round * sizeof(*sb));

        for (j = 0; j < round; j++) {
            sb[j] = malloc(size * sizeof(**sb));

            for (i = 0; i < size; i++) sb[j][i] = size - 1 - i;
        }

        /* send */
        for (j = 0; j < round; j++) {
            MPI_Isend(sb[j], size, MPI_INT, 1, j, comm, &req[j]);

            /* cancel */
            if (rand() % 2) MPI_Cancel(&req[j]);
        }

        /* wait */
        for (j = 0; j < round; j++) {
            MPI_Wait(&req[j], &status[j]);

            MPI_Test_cancelled(&status[j], &flag);

            if (flag) {
                printf("%d-th message cancelled\n", j);
            }
            else {
                printf("%d-th message sent\n", j);
            }
        }

        for (j = 0; j < round; j++) {
            free(sb[j]);
        }

        free(sb);
    }
    else if (rank == 1) {
        rb = calloc(size, sizeof(*rb));

        for (j = 0; j < round; j++) {
            rb[j] = calloc(size, sizeof(**rb));
        }

        /* dead lock */
        for (j = 0; j < round; j++) {
            MPI_Recv(rb[j], size, MPI_INT, 0, MPI_ANY_TAG, comm, &status[j]);
            printf("received, source: %d, tag: %d\n", status[j].MPI_SOURCE, status[j].MPI_TAG);
        }

        for (j = 0; j < round; j++) {
            free(rb[j]);
        }

        free(rb);
    }

end:
    MPI_Finalize();

    return 0;
}
