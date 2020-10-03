
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    int flag;

    /* test if MPI has been initialized or not */
    MPI_Initialized(&flag);

    if (flag) {
        printf("MPI is initialized.\n");
    }
    else {
        printf("MPI isn't initialized yet.\n");
    }

    /* initialize */
    MPI_Init(&argc, &argv);

    MPI_Initialized(&flag);

    if (flag) {
        printf("MPI is initialized.\n");
    }
    else {
        printf("MPI isn't initialized yet.\n");
    }

    printf("Hello World!\n");

    MPI_Finalize();

    return 0;
}
