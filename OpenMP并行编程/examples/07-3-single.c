
#include <stdio.h>

int main()
{
#pragma omp parallel
    {
#pragma omp single
        printf("Hello World!\n");
   }

    return 0;
}
