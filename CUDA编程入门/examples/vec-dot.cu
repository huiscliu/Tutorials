
/* dot product of two vectors: d = <x, y> */

#include "aux.h"
#include <assert.h>

typedef double FLOAT;

/* host, add */
FLOAT dot_host(FLOAT *x, FLOAT *y, int N)
{
    int i;
    FLOAT t = 0;

    assert(x != NULL);
    assert(y != NULL);

    for (i = 0; i < N; i++) t += x[i] * y[i];

    return t;
}

__device__ void warpReduce(volatile FLOAT *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

/* partial dot product */
__global__ void dot_stg_1(const FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    __shared__ FLOAT sdata[256];
    int idx = get_tid();
    int tid = threadIdx.x;
    int bid = get_bid();

    /* load data to shared mem */
    if (idx < N) {
        sdata[tid] = x[idx] * y[idx];
    }
    else {
        sdata[tid] = 0;
    }

    __syncthreads();

    /* reduction using shared mem */
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();

    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) z[bid] = sdata[0];
}

/* sum all entries in x and asign to y
 * block dim must be 256 */
__global__ void dot_stg_2(const FLOAT *x, FLOAT *y, int N)
{
    __shared__ FLOAT sdata[256];
    int idx = get_tid();
    int tid = threadIdx.x;
    int bid = get_bid();

    /* load data to shared mem */
    if (idx < N) {
        sdata[tid] = x[idx];
    }
    else {
        sdata[tid] = 0;
    }

    __syncthreads();

    /* reduction using shared mem */
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();

    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) y[bid] = sdata[0];
}

__global__ void dot_stg_3(FLOAT *x, int N)
{
    __shared__ FLOAT sdata[128];
    int tid = threadIdx.x;
    int i;

    sdata[tid] = 0;

    /* load data to shared mem */
    for (i = 0; i < N; i += 128) {
        if (tid + i < N) sdata[tid] += x[i + tid];
    }

    __syncthreads();

    /* reduction using shared mem */
    if (tid < 64) sdata[tid] = sdata[tid] + sdata[tid + 64];
    __syncthreads();

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) x[0] = sdata[0];
}

/* dz and d serve as cache: result stores in d[0] */
void dot_device(FLOAT *dx, FLOAT *dy, FLOAT *dz, FLOAT *d, int N)
{
    /* 1D block */
    int bs = 256;

    /* 2D grid */
    int s = ceil(sqrt((N + bs - 1.) / bs));
    dim3 grid = dim3(s, s);
    int gs = 0;

    /* stage 1 */
    dot_stg_1<<<grid, bs>>>(dx, dy, dz, N);

    /* stage 2 */
    {
        /* 1D grid */
        int N2 = (N + bs - 1) / bs;

        int s2 = ceil(sqrt((N2 + bs - 1.) / bs));
        dim3 grid2 = dim3(s2, s2);

        dot_stg_2<<<grid2, bs>>>(dz, d, N2);

        /* record gs */
        gs = (N2 + bs - 1.) / bs;
    }

    /* stage 3 */
    dot_stg_3<<<1, 128>>>(d, gs);
}

int main(int argc, char **argv)
{
    int N = 10000070;
    int nbytes = N * sizeof(FLOAT);

    FLOAT *hx = NULL, *hy = NULL;
    FLOAT *dx = NULL, *dy = NULL, *dz = NULL, *d = NULL;
    int i, itr = 20;
    FLOAT asd = 0, ash;
    double td, th;

    if (argc == 2) {
        int an;

        an = atoi(argv[1]);
        if (an > 0) N = an;
    }

    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);

    cudaMalloc((void **)&dz, sizeof(FLOAT) * ((N + 255) / 256));
    cudaMalloc((void **)&d, sizeof(FLOAT) * ((N + 255) / 256));

    if (dx == NULL || dy == NULL || dz == NULL || d == NULL) {
        printf("couldn't allocate GPU memory\n");
        return -1;
    }

    printf("allocated %e MB on GPU\n", nbytes / (1024.f * 1024.f));

    /* alllocate CPU mem */
    hx = (FLOAT *) malloc(nbytes);
    hy = (FLOAT *) malloc(nbytes);

    if (hx == NULL || hy == NULL) {
        printf("couldn't allocate CPU memory\n");
        return -2;
    }
    printf("allocated %e MB on CPU\n", nbytes / (1024.f * 1024.f));

    /* init */
    for (i = 0; i < N; i++) {
        hx[i] = 1;
        hy[i] = 2;
    }

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);

    /* let dust fall */
    cudaDeviceSynchronize();
    td = get_time();

    /* call GPU */
    for (i = 0; i < itr; i++) dot_device(dx, dy, dz, d, N);

    /* let GPU finish */
    cudaDeviceSynchronize();
    td = get_time() - td;

    th = get_time();
    for (i = 0; i < itr; i++) ash = dot_host(hx, hy, N);
    th = get_time() - th;

    /* copy data from GPU */
    cudaMemcpy(&asd, d, sizeof(FLOAT), cudaMemcpyDeviceToHost);

    printf("dot, answer: %d, calculated by GPU:%f, calculated by CPU:%f\n", 2 * N, asd, ash);
    printf("GPU time: %e, CPU time: %e, speedup: %g\n", td, th, th / td);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    cudaFree(d);

    free(hx);
    free(hy);

    return 0;
}
