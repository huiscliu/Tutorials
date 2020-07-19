
#include "utils.h"
#include "api_ftn.h"

int main(void)
{
    double x = 3;

    /* c */
    sls_print();

    /* fortran */
    sls_input_(&x);

    return 0;
}
