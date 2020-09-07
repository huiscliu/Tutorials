
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define abs(x)   (x<0 ? -x : x)
#define N        1000000

// return the smallest magnitude among all the integers in data[N]
int main()
{
    int i;
    int result = 0;
    int data[N];

#pragma omp parallel for
    for (i = 0; i < N; i++) data[i] = i;

#pragma omp declare reduction(maxabs : int :              \
        omp_out = abs(omp_in) < abs(omp_out) ? omp_out : abs(omp_in))\
        initializer (omp_priv=0)

#pragma omp parallel for reduction(maxabs:result)
    for (i = 0; i < N; i++) {
        if (abs(data[i]) > abs(result)) {
            result = abs(data[i]);
        }
    }

    printf("result: %d\n", result);

    return 0;
}
