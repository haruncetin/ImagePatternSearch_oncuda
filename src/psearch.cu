#include "include/psearch.cuh"

__device__ int check_pattern(uint16_t *I, unsigned int N, uint16_t *P, unsigned int K, int i, int j){

    int threshold = 5;

    // Find the center of the kernels
    int kc = K / 2;

    // Search for the pattern in the given region of the image matrix
    int match = 1;
    for (int m = 0; m < K; m++) {
        for (int n = 0; n < K; n++) {
            int ii = i + (m - kc);
            int jj = j + (n - kc);

            // Check if the input element is within the matrix bounds
            if (ii >= 0 && ii < N && jj >= 0 && jj < N) {
                int diff = _abs(I[ii * N + jj] - P[m * K + n]);
                if (diff > threshold) {
                    match = 0;
                    break;
                }
            } 
        }
    }
    return match;
}

__global__ void psearch_kernel(uint16_t *I, unsigned int N, uint16_t *P, unsigned int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Find the center of the kernels
    int kc = K / 2;

    // Check the input element is within the lower triangle or 
    // upper triangle of the image matrix
    if(i >= j) {
        // Search for the pattern in the lower triangle of the image matrix
        int matchInLowerTriangle = check_pattern(I, N, P, K, i, j);
        if (matchInLowerTriangle) {
            printf("Found in LOWER TRIANGLE at row: %d, col: %d\n", i-kc, j-kc);
        }
    } else {
        // Search for the pattern in the upper triangle of the image matrix
        int matchInUpperTriangle = check_pattern(I, N, P, K, i, j);
        if (matchInUpperTriangle) {
            printf("Found in UPPER TRIANGLE at row: %d, col: %d\n", i-kc, j-kc);
        }
    }
}
