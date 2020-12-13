
#include <stdio.h>
#include <omp.h>

int main()
{
    #pragma omp parallel num_threads(6)
    {
        int id = omp_get_thread_num();

        if (id == 0) printf("Coding\n");
        if (id == 1) printf("is\n");
        if (id == 2) printf("not\n");
        if (id == 3) printf("fun\n");
        if (id == 4) printf("at\n");
        if (id == 5) printf("all\n");
    }

    return 0;
}
