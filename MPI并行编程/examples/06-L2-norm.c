
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* rank 0 computes L2 norm of a vector
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

    /* rank 0 receives partial norm from other MPIs */
    if (rank == 0) {
        double temp;

#if 0
        for (i = 1; i < nprocs; i++) {
            MPI_Recv(&temp, 1, MPI_DOUBLE, i, 2020, comm, &status);

            nrm += temp;
        }
#else
        for (i = 1; i < nprocs; i++) {
            MPI_Recv(&temp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);

            nrm += temp;
        }
#endif

        nrm = sqrt(nrm);
        printf("L2 norm is %g\n", nrm);
    }
    else {
        MPI_Send(&nrm, 1, MPI_DOUBLE, 0, 2020, comm);
    }

    free(x);
    MPI_Finalize();

    return 0;
}
