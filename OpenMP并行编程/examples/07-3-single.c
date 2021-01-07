
#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp parallel
    {
#pragma omp single
        {
            printf("Hello, id: %d\n", omp_get_thread_num());
        }
    }

    return 0;
}
