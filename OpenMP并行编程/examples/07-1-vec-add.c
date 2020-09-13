
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <sys/time.h>
#include <time.h>

double omp_get_time(void)
{
    struct timeval tv;
    double t;

    gettimeofday(&tv, (struct timezone *)0);
    t = tv.tv_sec + (double)tv.tv_usec * 1e-6;

    return t;
}

int main(int argc, char **argv)
{
    int i, len = 100000000;
    double *x, *y, *z;
    double va_tm;

    if (argc == 2) {
        int tlen = atoi(argv[1]);

        if (tlen > 0) len = tlen;
    }

    /* malloc memory, no check */
    x = malloc(sizeof(*x) * len);
    y = malloc(sizeof(*y) * len);
    z = malloc(sizeof(*z) * len);

    va_tm = omp_get_time();

    #pragma omp parallel
    #pragma omp for
    for (i = 0; i < len; i++) {
        x[i] = i + 0.3;
        y[i] = i + M_PI;
    }

    #pragma omp parallel for
    for (i = 0; i < len; i++) {
        z[i] = x[i] + y[i];
    }

    va_tm = omp_get_time() - va_tm;
    printf("time: %g s\n", va_tm);

    free(x);
    free(y);
    free(z);

    return 0;
}
