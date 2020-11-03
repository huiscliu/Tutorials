
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int i, rank, nprocs;
    int rcv_count, total_count = 0;
    double *snd_buf = NULL;
    int *snd_count = NULL;
    int *snd_disps = NULL;
    double *rcv_buf = NULL;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* rand */
    srand(rank);

    if (rank == 0) {
        for (i = 0; i < nprocs; i++) {
            snd_count[i] = 1 + rand() % 10;
            printf("rank 0 will send %d data to rank %d\n", snd_count[i], i);
        }

        snd_disps[0] = 0;

        for (i = 1; i < nprocs; i++) {
            snd_disps[i] = snd_disps[i - 1] + snd_count[i - 1];
        }

        /* total length */
        total_count = snd_disps[nprocs - 1] + snd_count[nprocs - 1];
        snd_buf = malloc(total_count * sizeof(*snd_buf));

        for (i = 0; i < total_count; i++) snd_buf[i] = i;
    }

    MPI_Scatter(snd_count, 1, MPI_INT, &rcv_count, 1, MPI_INT, 0, comm);

    /* recv */
    rcv_buf = malloc(rcv_count * sizeof(*rcv_buf));
    printf("rank %d receives %d data\n", rank, rcv_count);

    MPI_Scatterv(snd_buf, snd_count, snd_disps, MPI_DOUBLE, rcv_buf, rcv_count, MPI_DOUBLE, 0, comm);

    free(snd_buf);
    free(rcv_buf);
    free(snd_count);
    free(snd_disps);

    MPI_Finalize();
    return 0;
}
