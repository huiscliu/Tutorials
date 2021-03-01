#include <stdio.h>
#include <cuda.h>

/* cg */
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void kernel()
{
    int oe = threadIdx.x & 1;  /* odd or even */

    /* include all threads */
    coalesced_group g1 = coalesced_threads();
    unsigned int mask = __activemask();

    if (oe) {
        coalesced_group g = coalesced_threads();
        mask = __activemask();
    }
    else {
        coalesced_group g = coalesced_threads();
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
