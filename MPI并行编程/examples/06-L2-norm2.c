
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* rank 0 computes L2 norm of a vector
 * Then rank 0 sends results to all other ranks
 *
 * length is 1 million
 * v[i] = i 
 *
 * L2 norm of a vector:
 * |v|_2 = sqrt(v[0]*v[0] + v[1]*v[1] + .... + v[n]*v[n]) 
 *       = (v[0]^2 + v[1]^2 + .... + v[n]^2)^0.5
 *
 * */
int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;

    int size = 1000000;
    double *x = NULL;
    int lb;    /* lower bound */
    int ub;    /* upper bound */
    int lsize; /* local work size */
    int i;
    double nrm = 0.;

    MPI_Status status;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    /* vector index distribution */
    lsize = size / nprocs;
    lb = lsize * rank;
    ub = lsize * (rank + 1);
    if (rank == nprocs - 1) ub = size;

    /* local work */
    lsize = ub - lb;
    x = malloc(lsize * sizeof(*x));
    for (i = 0; i < lsize; i++) x[i] = i + lb;

    /* partial norm */
    for (i = 0; i < lsize; i++) nrm += x[i] * x[i];

    if (rank == 0) {
        double temp;

        /* rank 0 receive partial L2 norms from other ranks */
        for (i = 1; i < nprocs; i++) {
            MPI_Recv(&temp, 1, MPI_DOUBLE, i, 2020, comm, &status);

            nrm += temp;
        }

        nrm = sqrt(nrm);

        /* rank 0 sends final results to all MPIs */
        for (i = 1; i < nprocs; i++) {
            MPI_Send(&nrm, 1, MPI_DOUBLE, i, 202010, comm);
        }

        printf("L2 norm is %g\n", nrm);
    }
    else {
        /* sends partial norm to rank 0 */
        MPI_Send(&nrm, 1, MPI_DOUBLE, 0, 2020, comm);

        /* receive final result from rank 0 */
        MPI_Recv(&nrm, 1, MPI_DOUBLE, 0, 202010, comm, &status);
    }

    printf("rank: %d, L2 norm: %g\n", rank, nrm);

    free(x);
    MPI_Finalize();

    return 0;
}
