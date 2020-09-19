
#include "utils.h"
#include "add.h"

int main(void)
{
    double x, y;

    /* print */
    sls_print();

    /* sum */
    x = 1;
    y = 3.14;

    printf("sum of %g and %g is: %g\n", x, y, sls_add(x, y));

    return 0;
}
