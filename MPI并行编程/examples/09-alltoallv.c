
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int i, rank, nprocs, size;
    int *rb = NULL;
    int *sb = NULL;
    int *snd_cnt = NULL, *snd_dis = NULL;
    int *rcv_cnt = NULL, *rcv_dis = NULL;

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    snd_cnt = malloc(nprocs * sizeof(*snd_cnt));
    rcv_cnt = malloc(nprocs * sizeof(*rcv_cnt));

    snd_dis = malloc(nprocs * sizeof(*snd_dis));
    rcv_dis = malloc(nprocs * sizeof(*rcv_dis));

    /* random */
    srand(rank);

    /* send counts */
    size = 0;
    for (i = 0; i < nprocs; i++) {
        snd_cnt[i] = 1 + rand() % 10;
        size += snd_cnt[i];
    }

    /* send displs */
    snd_dis[0] = 0;
    for (i = 1; i < nprocs; i++) snd_dis[i] = snd_dis[i - 1] + snd_cnt[i - 1];

    sb = malloc(size * sizeof(*sb));
    for (i = 0; i < size; i++) sb[i] = rand();

    /* receive */
    MPI_Alltoall(snd_cnt, 1, MPI_INT, rcv_cnt, 1, MPI_INT, comm);

    /* send displs */
    rcv_dis[0] = 0;
    size = rcv_cnt[0];
    for (i = 1; i < nprocs; i++) {
        rcv_dis[i] = rcv_dis[i - 1] + rcv_cnt[i - 1];
        size += rcv_cnt[i];
    }

    /* buffers */
    rb = malloc(size * sizeof(*rb));

    /* alltoalv */
    MPI_Alltoallv(sb, snd_cnt, snd_dis, MPI_INT, rb, rcv_cnt, rcv_dis, MPI_INT, comm);

    free(rb);
    free(sb);
    free(snd_cnt);
    free(rcv_cnt);
    free(snd_dis);
    free(rcv_dis);

    MPI_Finalize();
    return 0;
}

