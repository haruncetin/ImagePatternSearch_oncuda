#include "include/psearch.cuh"

int main(int argc, char **argv) {
    printf("Running...\n");

    unsigned int I_width, I_height, P_width, P_height;
    uint16_t *I, *P;

    if (argc != 3) {
        printf("Usage: %s original.bmp pattern.bmp\n", argv[0]);
        exit(0);
    }

    printf("Reading images...\n");

    I = read_png_1d(argv[1], &I_width, &I_height);
    printf("I_width: %d, I_height: %d\n", I_width, I_height);
    P = read_png_1d(argv[2], &P_width, &P_height);
    printf("A_width: %d, A_height: %d\n", P_width, P_height);

    if (I == 0 || P == 0) {
        printf("Error: Failed to read the image.\n");
        exit(1);
    }

    if (I_width < P_width || I_height < P_height) {
        fprintf(stderr, "Error: The pattern cannot be larger than the picture\n");
        exit(EXIT_FAILURE);
    }

    // The device pointers
    uint16_t *d_I, *d_P;

    // Allocate memory on the device
    printf("Allocating device memory...\n");
    cudaMalloc(&d_I, I_width * I_height * sizeof(uint16_t));
    cudaMalloc(&d_P, P_width * P_height * sizeof(uint16_t));

    // Copy the input matrix to the device
    printf("Copying to device...\n");
    cudaMemcpy(d_I, I, I_width * I_height * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, P, P_width * P_height * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    printf("Launching kernel...\n");
    dim3 blockSize(32, 32);
    dim3 gridSize((I_width + blockSize.x - 1) / blockSize.x, (I_height + blockSize.y - 1) / blockSize.y);

    printf("Grid size: %d, %d\n", gridSize.x, gridSize.y);
    printf("Block size: %d, %d\n", blockSize.x, blockSize.y);

    printf("Searching for the pattern with CUDA...\n");

    // Create CUDA events for measuring performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start time
    cudaEventRecord(start);

    // Launch the kernel
    psearch_kernel<<<gridSize, blockSize>>>(d_I, I_width, d_P, P_width);
    
    // Record the end time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %.8f ms\n", milliseconds);

    // Free the device memory
    cudaFree(d_I);
    cudaFree(d_P);

    free(I);
    free(P);

    return 0;
}