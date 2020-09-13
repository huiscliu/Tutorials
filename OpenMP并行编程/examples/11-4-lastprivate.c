
#include <omp.h>
#include <stdio.h>

int main()
{
    int n = 0;
    int i;

    #pragma omp parallel for lastprivate(n)
    for (i = 0; i < 20; i++) {
        int id;

        id = omp_get_thread_num();

        n = -i;
        printf("This is thread: %d. Value of n is changed to: %d\n", id, n);
    }

    printf("\n\n\n");
    printf("Final n is: %d\n", n);

    return 0;
}
