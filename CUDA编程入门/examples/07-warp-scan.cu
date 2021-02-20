#include <stdio.h>
#include <cuda.h>

__global__ void scan4()
{
    int laneId = threadIdx.x & 0x1f;
    // Seed sample starting value (inverse of lane ID)
    int value = 31 - laneId;

    // Loop to accumulate scan within my partition.
    // Scan requires log2(n) == 3 steps for 8 threads
    // It works by an accumulated sum up the warp
    // by 1, 2, 4, 8 etc. steps.
    for (int i=1; i<=4; i*=2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        int n = __shfl_up_sync(0xffffffff, value, i, 8);

        if ((laneId & 7) >= i) value += n;
    }

    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main()
{
    scan4<<< 1, 32 >>>();
    cudaDeviceSynchronize();

    return 0;
}
