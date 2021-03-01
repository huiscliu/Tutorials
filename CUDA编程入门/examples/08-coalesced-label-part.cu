#include <stdio.h>
#include <cuda.h>

/* cg */
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void kernel()
{
    int oe = threadIdx.x % 5;  /* 0, 1, 2, 3, 4 */

    /* include all threads */
    coalesced_group g1 = coalesced_threads();
    unsigned int mask = __activemask();

    /* g2 and g3 are equivalent */
    coalesced_group g2 = labeled_partition(g1, oe);

    if (oe == 0) {
        coalesced_group g3 = coalesced_threads();
        mask = __activemask();
    }
    else if (oe == 1) {
        coalesced_group g3 = coalesced_threads();
        mask = __activemask();
    }
    else if (oe == 2) {
        coalesced_group g3 = coalesced_threads();
        mask = __activemask();
    }
    else if (oe == 3) {
        coalesced_group g3 = coalesced_threads();
        mask = __activemask();
    }
    else {
        coalesced_group g3 = coalesced_threads();
        mask = __activemask();
    }

    printf("Thread %d final mask = %ud\n", threadIdx.x, mask);
}

int main()
{
    kernel<<< 1, 32 >>>();
    cudaDeviceSynchronize();

    return 0;
}
