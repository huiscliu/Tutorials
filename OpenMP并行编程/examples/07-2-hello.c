
#include <stdio.h>
#include <omp.h>

int main()
{
    #pragma omp parallel
    #pragma omp sections
    {
        #pragma omp section
        printf("Hello, id: %d\n", omp_get_thread_num());

        #pragma omp section
        printf("Hi, id: %d\n", omp_get_thread_num());

        #pragma omp section
        printf("Nihao, id: %d\n", omp_get_thread_num());

        #pragma omp section
        printf("Bonjour, id: %d\n", omp_get_thread_num());

        #pragma omp section
        printf("Kon'nichiwa, id: %d\n", omp_get_thread_num());
    }

    return 0;
}
