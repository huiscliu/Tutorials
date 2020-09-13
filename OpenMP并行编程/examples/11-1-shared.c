
#include <omp.h>
#include <stdio.h>

int main()
{
    int n = 20;

    #pragma omp parallel
    {
        int id;

        id = omp_get_thread_num();

        printf("This is thread: %d. I can see shared variable n: %d\n", id, n);
    }

    printf("\n\n\n");

    #pragma omp parallel
    {
        int id;

        id = omp_get_thread_num();

        n = id;
        printf("This is thread: %d. I change n to my ID: %d\n", id, n);
    }

    printf("\n\n\n");
    printf("Data race happened and final n is: %d\n", n);

    return 0;
}
