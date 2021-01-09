
#include <omp.h>
#include <stdio.h>

int n = 20;

#pragma omp threadprivate(n)

int main()
{
    printf("n: %d\n", n);

    #pragma omp parallel
    {
        int id;

        id = omp_get_thread_num();

        n = id * 8;

        printf("This is thread: %d. Value of n is changed to: %d\n", id, n);
    }

    printf("Serial part n is: %d\n\n", n);

    #pragma omp parallel
    {
        int id;

        id = omp_get_thread_num();

        printf("This is thread: %d. Value of n is changed to: %d\n", id, n);
    }

    printf("Serial part n is: %d\n\n", n);

    printf("copyin\n");

    #pragma omp parallel copyin(n)
    {
        int id;

        id = omp_get_thread_num();

        printf("This is thread: %d. Value of n is changed to: %d\n", id, n);
    }

    printf("Serial part n is: %d\n\n", n);

    return 0;
}
