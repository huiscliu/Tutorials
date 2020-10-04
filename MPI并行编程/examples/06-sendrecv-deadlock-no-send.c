
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    int sb[4];
    int rb[4] = {0, 0, 0, 0};
    MPI_Status status;

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
        /* to make compiler happy */
        (void) sb;
#if 0
        MPI_Send(sb, 2, MPI_INT, 1, 999, comm);
#endif
    }
    else if (rank == 1) {
        MPI_Recv(rb, 2, MPI_INT, 0, 999, comm, &status);

        printf("received: %d %d\n", rb[0], rb[1]);
    }

end:
    MPI_Finalize();

    return 0;
}
