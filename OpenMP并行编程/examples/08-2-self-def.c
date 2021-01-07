
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define abs(x)   ((x) < 0 ? -(x) : (x))

int main()
{
    int i;
    int result = 0;

    int N = 20000;
    int data[20000];

    /* seed */
    srand(0);

    /* init */
    for (i = 0; i < N; i++) data[i] = ((i % 2) * 2 - 1) * rand();

    #pragma omp declare reduction(maxabs : int :              \
        omp_out = abs(omp_in) < abs(omp_out) ? omp_out : omp_in)\
        initializer (omp_priv=0)

    #pragma omp parallel for reduction(maxabs:result)
    for (i = 0; i < N; i++) {
        if (abs(data[i]) > abs(result)) {
            result = data[i];
        }
    }

    printf("result: %d\n", result);

    return 0;
}
