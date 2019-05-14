
/* asum: sum of all entries of a vector */

#include <stdio.h>
#include <cuda.h>

typedef double FLOAT;
#define USE_UNIX 1

/* get thread id: 1D block and 2D grid */
#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)

/* get block id: 2D grid */
#define get_bid() (blockIdx.x + blockIdx.y * gridDim.x)

/* warm up, start GPU, optional */
void warmup();

/* get time stamp */
double get_time(void);

/* asum host */
FLOAT asum_host(FLOAT *x, int N);

/* a little system programming */
#if USE_UNIX

#include <sys/time.h>
#include <time.h>

double get_time(void)
{
    struct timeval tv;
    double t;

    gettimeofday(&tv, (struct timezone *)0);
    t = tv.tv_sec + (double)tv.tv_usec * 1e-6;

    return t;
}
#else
#include <windows.h>

double get_time(void)
{
    LARGE_INTEGER timer;
    static LARGE_INTEGER fre;
    static int init = 0;
    double t;

    if (init != 1) {
        QueryPerformanceFrequency(&fre);
        init = 1;
    }

    QueryPerformanceCounter(&timer);

    t = timer.QuadPart * 1. / fre.QuadPart;

    return t;
}
#endif

/* warm up GPU */
__global__ void warmup_knl()
{
    int i, j;

    i = 1;
    j = 2;
    i = i + j;
}

void warmup()
{
    int i;

    for (i = 0; i < 8; i++) {
        warmup_knl<<<1, 256>>>();
    }
}

/* host, add */
FLOAT asum_host(FLOAT *x, int N)
{
    int i;
    FLOAT t = 0;

    for (i = 0; i < N; i++) t += x[i];

    return t;
}

/* sum all entries in x and asign to y
 * block dim must be 256 */
__global__ void asum_stg_1(const FLOAT *x, FLOAT *y, int N)
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

    if (tid < 32) sdata[tid] += sdata[tid + 32];
    __syncthreads();

    if (tid < 16) sdata[tid] += sdata[tid + 16];
    __syncthreads();

    if (tid < 8) sdata[tid] += sdata[tid + 8];
    __syncthreads();

    if (tid < 4) sdata[tid] += sdata[tid + 4];
    __syncthreads();

    if (tid < 2) sdata[tid] += sdata[tid + 2];
    __syncthreads();

    if (tid == 0) {
        y[bid] = sdata[0] + sdata[1];
    }
}

__global__ void asum_stg_3(FLOAT *x, int N)
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

    if (tid < 32) sdata[tid] = sdata[tid] + sdata[tid + 32];
    __syncthreads();

    if (tid < 16) sdata[tid] += sdata[tid + 16];
    __syncthreads();

    if (tid < 8) sdata[tid] += sdata[tid + 8];
    __syncthreads();

    if (tid < 4) sdata[tid] += sdata[tid + 4];
    __syncthreads();

    if (tid < 2) sdata[tid] += sdata[tid + 2];
    __syncthreads();

    if (tid == 0) {
        x[0] = sdata[0] + sdata[1];
    }
}

/* dy and dz serve as cache: result stores in dz[0] */
void asum(FLOAT *dx, FLOAT *dy, FLOAT *dz, int N)
{
    /* 1D block */
    int bs = 256;

    /* 2D grid */
    int s = ceil(sqrt((N + bs - 1.) / bs));
    dim3 grid = dim3(s, s);
    int gs = 0;

    /* stage 1 */
    asum_stg_1<<<grid, bs>>>(dx, dy, N);

    /* stage 2 */
    {
        /* 1D grid */
        int N2 = (N + bs - 1) / bs;

        int s2 = ceil(sqrt((N2 + bs - 1.) / bs));
        dim3 grid2 = dim3(s2, s2);

        asum_stg_1<<<grid2, bs>>>(dy, dz, N2);

        /* record gs */
        gs = (N2 + bs - 1.) / bs;
    }

    /* stage 3 */
    asum_stg_3<<<1, 128>>>(dz, gs);
}

int main()
{
    int N = 10000070;
    int nbytes = N * sizeof(FLOAT);

    FLOAT *dx = NULL, *hx = NULL;
    FLOAT *dy = NULL, *dz;
    int i, itr = 20;
    FLOAT asd = 0, ash;
    double td, th;

    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, sizeof(FLOAT) * ((N + 255) / 256));
    cudaMalloc((void **)&dz, sizeof(FLOAT) * ((N + 255) / 256));

    if (dx == NULL || dy == NULL || dz == NULL) {
        printf("couldn't allocate GPU memory\n");
        return -1;
    }

    printf("allocated %e MB on GPU\n", nbytes / (1024.f * 1024.f));

    /* alllocate CPU mem */
    hx = (FLOAT *) malloc(nbytes);

    if (hx == NULL) {
        printf("couldn't allocate CPU memory\n");
        return -2;
    }
    printf("allocated %e MB on CPU\n", nbytes / (1024.f * 1024.f));

    /* init */
    for (i = 0; i < N; i++) {
        hx[i] = 1;
    }

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

    /* let dust fall */
    cudaThreadSynchronize();
    td = get_time();

    /* call GPU */
    for (i = 0; i < itr; i++) asum(dx, dy, dz, N);

    /* let GPU finish */
    cudaThreadSynchronize();
    td = get_time() - td;

    th = get_time();
    for (i = 0; i < itr; i++) ash = asum_host(hx, N);
    th = get_time() - th;

    /* copy data from GPU */
    cudaMemcpy(&asd, dz, sizeof(FLOAT), cudaMemcpyDeviceToHost);

    printf("asum, answer: %d, calculated by GPU:%f, calculated by CPU:%f\n", N, asd, ash);
    printf("GPU time: %e, CPU time: %e, speedup: %g\n", td, th, th / td);

    cudaFree(dx);
    cudaFree(dy);
    free(hx);

    return 0;
}
