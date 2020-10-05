
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* computes L2 norm of a vector
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
    double nrm = 0., temp;

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

    /* reduce operation */
    MPI_Allreduce(&nrm, &temp, 1, MPI_DOUBLE, MPI_SUM, comm);

    nrm = sqrt(temp);
    printf("rank: %d, L2 norm: %g\n", rank, nrm);

    free(x);
    MPI_Finalize();

    return 0;
}
