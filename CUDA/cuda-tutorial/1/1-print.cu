#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void test01()
{
    // print the blocks and threads IDs
    // warp = 32 threads. (64 threads/block) --> ( 64/32 = 2 warps/block)
    int warp_ID_Value = 0;
    warp_ID_Value = threadIdx.x / 32;//%
    printf("The block ID is %d --- The thread ID is %d --- The warp ID %d\n",blockIdx.x,threadIdx.x, warp_ID_Value);
}

int main()
{   // Kernel launches 2 blocks with 64 threads each (total 128 threads)
    // Each block runs on an available SM (multiple blocks *can* run on same SM if resources allow)
    // Threads within a block are scheduled on the SM's CUDA cores (not 1:1 mapping to physical cores)
    // Each block contains 2 warps of 32 threads each (64 threads total per block)
    // Total: 4 warps across 2 blocks
    
    // Flow of Execution:
    // 1. Kernel Launch (`<<<2, 64>>>`): The CUDA runtime schedules the launch of 2 thread blocks, each containing 64 threads.
    // 2. Block Assignment: Each of the 2 blocks is assigned to an available Streaming Multiprocessor (SM) on the GPU [[7]].
    // 3. Warp Formation: Once a block is assigned to an SM, it is divided into 32-thread units called warps [[4]]. For this kernel, each block forms 2 distinct warps (warp 0: threads 0-31; warp 1: threads 32-63).
    // 4. Warp Scheduling: The SM's warp scheduler manages the execution of these warps. Warps are the smallest unit scheduled for execution, and their constituent threads execute in lockstep (SIMT). The SM can manage multiple warps concurrently, switching between them to hide latency.
    // 5. Thread Execution: Individual threads within an active warp execute the same instruction. The `printf` statement is executed once by each of the 128 threads, printing its unique blockIdx.x, threadIdx.x, and calculated warp ID.
    // 6. Synchronization: `cudaDeviceSynchronize()` ensures the host CPU waits until the entire kernel execution (all blocks and their threads) completes before proceeding.
    
    test01 <<<2, 64 >>> (); // Max number of threads per block is decided by the GPU architecture
    test01 <<<1, 8192 >>> (); // this wont cause build error but will not execute as it exceeds the maximum threads per block limit 
    // test01 <<<1, 1024 >>> (); // this is a valid. checkout white paper for "Max threads per block" for current GPU architectures. Similarly for "Max Thread blocks" 
    cudaDeviceSynchronize();
    return 0;
}