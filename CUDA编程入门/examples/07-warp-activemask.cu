#include <stdio.h>
#include <cuda.h>

__global__ void kernel()
{
    int oe = threadIdx.x & 0x1;  /* odd or even */
    unsigned int mask;

    /* sepatrate a warp to 2 sub-groups and generate masks */
    if (oe) {
        mask = __activemask();
    }
    else {
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
