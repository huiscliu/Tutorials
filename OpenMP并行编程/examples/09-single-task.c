
#include <omp.h>
#include <stdio.h>

void task_func(int id, int v)
{
    printf("id: %d, i: %d\n", id, v);
}

int main()
{
#pragma omp parallel
    {
        int id;
        int i;

        id = omp_get_thread_num();

#pragma omp single
        {
            for (i = 0; i < 20; i++) {
#pragma omp task
                task_func(id, i);
            }
        }
    }

    return 0;
}
