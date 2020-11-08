
#include <mpi.h>
#include <stdio.h>

typedef struct {
    int rank;
    int id;
    double weight;

} ELEM_INFO;

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Status status;
    MPI_Datatype newtype;
    ELEM_INFO sb = {0, 0, 1.3};
    ELEM_INFO rb = {1, 1, -2.3};
    int blk[3];
    MPI_Datatype type[3];
    MPI_Aint disp[3];

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if (nprocs < 2) {
        printf("At least 2 MPIs are required for this example\n");
        goto end;
    }

    /* new type, send first and forth elements */
    blk[0] = blk[1] = blk[2] = 1;
    type[0] = type[1] = MPI_INT;
    type[2] = MPI_DOUBLE;

    /* disp */
    MPI_Address(&sb.rank, &disp[0]);
    MPI_Address(&sb.id, &disp[1]);
    MPI_Address(&sb.weight, &disp[2]);

    disp[2] -= disp[0];
    disp[1] -= disp[0];
    disp[0] = 0;

    MPI_Type_struct(3, blk, disp, type, &newtype);

    MPI_Type_commit(&newtype);

    /* process 0 sends info to 1 */
    if (rank == 0) {
        MPI_Send(&sb, 1, newtype, 1, 999, comm);
    }
    else if (rank == 1) {
        printf("initial received: %d %d %g\n", rb.rank, rb.id, rb.weight);
        MPI_Recv(&rb, 1, newtype, 0, 999, comm, &status);
        printf("received: %d %d %g\n", rb.rank, rb.id, rb.weight);
    }

    MPI_Type_free(&newtype);

end:
    MPI_Finalize();

    return 0;
}
