
#include <stdio.h>

int main()
{
#pragma omp parallel
#pragma omp sections
    {
#pragma omp section
        printf("Hello!\n");

#pragma omp section
        printf("Hi!\n");

#pragma omp section
        printf("Nihao!\n");

#pragma omp section
        printf("Bonjour!\n");

#pragma omp section
        printf("Kon'nichiwa!\n");
    }

    return 0;
}
