
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    double *x, *y, d = 0.;
    int size = 32, i;

    x = malloc(size * sizeof(*x));
    y = malloc(size * sizeof(*y));

    for (i = 0; i < size; i++) d += x[i] * y[i];

    free(x);
    free(y);

    return 0;
}
