#include <stdio.h>
#include <cuda.h>

/* cg */
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void bcast(int arg) 
{
    int oe = threadIdx.x & 0x1;  /* odd or even */
    int value = -1;

    if (oe) { /* broadarg */
        coalesced_group g = coalesced_threads();
        int laneId = g.thread_rank();

        if (laneId == 0)  value = arg;

        // Synchronize all threads in warp, and get "value" from lane 0
        value = g.shfl(value, 0);

        if (value != arg) printf("Thread %d failed.\n", threadIdx.x);
    }
    else { /* broadcast arg + 1 */
        coalesced_group g = coalesced_threads();
        int laneId = g.thread_rank();

        if (laneId == 0)  value = arg + 1;

        // Synchronize all threads in warp, and get "value" from lane 0
        value = g.shfl(value, 0);

        if (value != arg + 1) printf("Thread %d failed.\n", threadIdx.x);
    }
}

int main()
{
    bcast<<< 1, 32 >>>(1234);
    cudaDeviceSynchronize();

    return 0;
}
