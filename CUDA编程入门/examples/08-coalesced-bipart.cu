#include <stdio.h>
#include <cuda.h>

/* cg */
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void kernel()
{
    int oe = threadIdx.x & 0x1;  /* odd or even */

    /* include all threads */
    coalesced_group g1 = coalesced_threads();
    unsigned int mask = __activemask();

    /* g2, g3 and g4 are equivalent */
    coalesced_group g2 = binary_partition(g1, oe);
    coalesced_group g3 = labeled_partition(g1, oe);

    if (oe) {
        coalesced_group g4 = coalesced_threads();
        mask = __activemask();
    }
    else {
        coalesced_group g4 = coalesced_threads();
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
