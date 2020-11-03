
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int i, rank, nprocs;
    int snd_count, total_count = 0;
    double *snd_buf = NULL;
    int *rcv_count = NULL;
    int *rcv_disps = NULL;
    double *rcv_buf = NULL;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* rand */
    srand(rank);

    /* send */
    snd_count = 1 + rand() % 20;
    snd_buf = malloc(snd_count * sizeof(*snd_buf));

    for (i = 0; i < snd_count; i++) snd_buf[i] = rand() % 3;

    printf("rank %d sends %d data\n", rank, snd_count);

    /* sync */
    MPI_Barrier(comm);

    /* recv */
    rcv_count = malloc(nprocs * sizeof(*rcv_count));
    rcv_disps = malloc(nprocs * sizeof(*rcv_disps));

    MPI_Allgather(&snd_count, 1, MPI_INT, rcv_count, 1, MPI_INT, comm);

    rcv_disps[0] = 0;
    for (i = 1; i < nprocs; i++) {
        rcv_disps[i] = rcv_disps[i - 1] + rcv_count[i - 1];
    }

    /* total length */
    total_count = rcv_disps[nprocs - 1] + rcv_count[nprocs - 1];
    rcv_buf = malloc(total_count * sizeof(*rcv_buf));

    /* gatherv */
    MPI_Allgatherv(snd_buf, snd_count, MPI_DOUBLE, rcv_buf, rcv_count, rcv_disps, MPI_DOUBLE, comm);

    free(snd_buf);
    free(rcv_buf);
    free(rcv_count);
    free(rcv_disps);

    MPI_Finalize();
    return 0;
}

