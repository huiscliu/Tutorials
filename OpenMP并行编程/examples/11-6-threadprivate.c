
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

    printf("\n\n\n");
    printf("Final n is: %d\n", n);

    #pragma omp parallel
    {
        int id;

        id = omp_get_thread_num();

        printf("This is thread: %d. Value of n is changed to: %d\n", id, n);
    }

    printf("\n\n\n");
    printf("Final n is: %d\n", n);

    return 0;
}
