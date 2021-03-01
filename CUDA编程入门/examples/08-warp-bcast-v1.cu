#include <stdio.h>
#include <cuda.h>

/* cg */
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void bcast(int arg) 
{
    coalesced_group g = coalesced_threads();
    int laneId = g.thread_rank();
    int value;

    if (laneId == 0)  value = arg; 

    // Synchronize all threads in warp, and get "value" from lane 0
    value = g.shfl(value, 0);

    if (value != arg) printf("Thread %d failed.\n", threadIdx.x);
}

int main()
{
    bcast<<< 1, 32 >>>(1234);
    cudaDeviceSynchronize();

    return 0;
}
