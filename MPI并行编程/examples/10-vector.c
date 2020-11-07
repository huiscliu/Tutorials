
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    int sb[6] = {1, 1, 1, 1, 1, 1};
    int rb[6] = {0, 0, 0, 0, 0, 0};
    MPI_Status status;
    MPI_Datatype newtype;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if (nprocs < 2) {
        printf("At least 2 MPIs are required for this example\n");
        goto end;
    }

    /* new type, send first and forth elements */
    MPI_Type_vector(2, 1, 3, MPI_INT, &newtype);
    MPI_Type_commit(&newtype);

    /* process 0 sends info to 1 */
    if (rank == 0) {
        MPI_Send(sb, 1, newtype, 1, 999, comm);
    }
    else if (rank == 1) {
        MPI_Recv(rb, 1, newtype, 0, 999, comm, &status);

        printf("received: %d %d %d %d %d %d\n", rb[0], rb[1], rb[2], rb[3], rb[4], rb[5]);
    }

    MPI_Type_free(&newtype);

end:
    MPI_Finalize();

    return 0;
}
