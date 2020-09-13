
#include <stdio.h>
#include <omp.h>

int main()
{
    omp_set_num_threads(3);

    #pragma omp parallel
    {
        printf("Hello World!\n");
    }

    return 0;
}
