
#include <stdio.h>
#include <omp.h>

int main()
{
    printf("serial part, total number of threads: %d\n\n", omp_get_num_threads());

    #pragma omp parallel
    {
        printf("Hello World, total number of threads: %d\n", omp_get_num_threads());
    }

    return 0;
}
