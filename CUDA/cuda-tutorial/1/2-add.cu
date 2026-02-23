#include <stdio.h>
#include <cuda_runtime.h>

// #define SIZE 1024
#define SIZE 1024*1024*32  // Define the size of the vectors

// CUDA Kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Declare host pointers. These variables hold the ADDRESS of integers in host memory.
    int *A, *B, *C;            // A, B, C are pointers to integers on the host (CPU side)
    
    // Declare device pointers. These variables will hold the ADDRESS of integers in GPU memory.
    int *d_A, *d_B, *d_C;      // d_A, d_B, d_C are pointers to integers on the device (GPU side)
    
    int size = SIZE * sizeof(int); // Calculate the total memory size needed in bytes

    // CUDA event creation, used for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); // & gets the address of the 'start' variable
    cudaEventCreate(&stop);  // & gets the address of the 'stop' variable

    // Allocate and initialize host vectors (CPU memory)
    // malloc returns an ADDRESS where memory was allocated. We assign this ADDRESS to our pointer.
    A = (int *)malloc(size); // malloc gives us an address; assign it to A
    B = (int *)malloc(size); // malloc gives us an address; assign it to B
    C = (int *)malloc(size); // malloc gives us an address; assign it to C
    for (int i = 0; i < SIZE; i++) {
        A[i] = i;         // Use A like an array (it points to the start of the allocated memory)
        B[i] = SIZE - i;  // Use B like an array
    }

    // Allocate device vectors (GPU memory)
    // cudaMalloc needs to store the ADDRESS it allocates into the variable passed to it.
    // It needs the ADDRESS of the pointer variable itself (e.g., &d_A) to modify it.
    // Therefore, cudaMalloc expects a pointer-to-a-pointer (int**) as its first argument.
    cudaMalloc((void **)&d_A, size); // &d_A gets the address of the pointer variable d_A. **d_A means "pointer to a pointer".
    cudaMalloc((void **)&d_B, size); // &d_B gets the address of the pointer variable d_B.
    cudaMalloc((void **)&d_C, size); // &d_C gets the address of the pointer variable d_C.

    // Copy host vectors to device (from CPU memory to GPU memory)
    // cudaMemcpy copies data from one memory location to another.
    // The first two arguments are addresses: destination (d_A) and source (A).
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); // Copy data from host A to device d_A
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice); // Copy data from host B to device d_B

    // Start recording
    cudaEventRecord(start);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 96;
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock; // Ceiling division
    // Pass the ADDRESS stored in the device pointers (d_A, d_B, d_C) to the kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, SIZE);

    // vectorAdd<<<1, 1024>>>(d_A, d_B, d_C, SIZE);

    // Stop recording
    cudaEventRecord(stop);

    // Copy result back to host (from GPU memory to CPU memory)
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost); // Copy data from device d_C to host C

    // Calculate and print the execution time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f milliseconds\n", milliseconds);

    // Cleanup (free memory)
    cudaFree(d_A); // Free the GPU memory pointed to by d_A
    cudaFree(d_B); // Free the GPU memory pointed to by d_B
    cudaFree(d_C); // Free the GPU memory pointed to by d_C
    free(A);       // Free the CPU memory pointed to by A
    free(B);       // Free the CPU memory pointed to by B
    free(C);       // Free the CPU memory pointed to by C
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}