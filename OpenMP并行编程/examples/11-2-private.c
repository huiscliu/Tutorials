
#include <omp.h>
#include <stdio.h>

int main()
{
    int n = 20;

    #pragma omp parallel private(n)
    {
        int id;

        id = omp_get_thread_num();

        /* n is private and NOT initialized.
         * Compiler won't be happy */
        printf("This is thread: %d. Initial value of n is: %d\n", id, n);

        n = -id;
        printf("This is thread: %d. Value of n is: %d\n", id, n);
    }

    printf("\n\n\n");
    printf("Final n is: %d\n", n);

    return 0;
}
