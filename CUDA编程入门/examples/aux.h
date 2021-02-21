
#ifndef REDUCTION_AUX
#define REDUCTION_AUX

#include <stdio.h>
#include <cuda.h>

/* get thread id: 1D block and 2D grid */
#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)

/* get block id: 2D grid */
#define get_bid() (blockIdx.x + blockIdx.y * gridDim.x)

/* get time stamp */
double get_time(void);

/* a little system programming */
#if defined(__linux__) || defined(__APPLE__)

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

#elif defined(_WIN32)

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
#else
#error "unknown OS"
#endif

#endif
