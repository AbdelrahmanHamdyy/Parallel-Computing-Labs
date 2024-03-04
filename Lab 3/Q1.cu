#include <stdio.h>
#include <stdlib.h>

// Define the block dimension
#define NUM_THREADS 1024

// Define the data type for the array elements
typedef long long ll;

__global__ void parallelArraySum(double *input, double *output, ll n) {
    // Declare shared memory for storing each complete addition by all threads
    __shared__ double accumulator[NUM_THREADS];

    // Thread ID within the block
    ll tid = threadIdx.x;
    
    // Initialize sum to 0
    double sum = 0;
    
    // Each thread accumulates sum of its assigned elements
    for (ll i = tid; i < n; i += NUM_THREADS)
        sum += input[i];
    
    // Store the partial sum computed by this thread into shared memory
    accumulator[tid] = sum;
    
    // Synchronize threads within the block to ensure all partial sums are stored
    __syncthreads();

    // Perform parallel reduction within the block
    for (ll stride = NUM_THREADS / 2; stride > 0; stride /= 2) {
        // Each thread adds its sum with the partial sum at an offset
        if (tid < stride)
            accumulator[tid] += accumulator[tid + stride];
        // Synchronize threads to ensure all additions are complete before moving to next iteration
        __syncthreads();
    }

    // Store the final sum (at index 0) to the output array
    if (tid == 0)
        output[0] = accumulator[0];
}

int main(int argc, char *argv[]) {
    // Open the file for reading
    FILE *file = fopen(argv[1], "r");

    // Check if file was opened successfully
    if (!file) {
        fprintf(stderr, "Error opening file.\n");
        exit(1);
    }

    // Read the number of elements from the file
    ll n = 0;
    double val; // Temporary variable to store each value read from the file
    while (fscanf(file, "%lf", &val) == 1)
        n++; // Increment the count of elements as long as a valid floating-point number is read

    // Reset file pointer to beginning
    fseek(file, 0, SEEK_SET);

    // Allocate host memory for the array
    double *h_input = (double*)malloc(sizeof(double) * n);

    // Read array elements from the file
    for (ll i = 0; i < n; i++)
        fscanf(file, "%lf", &h_input[i]);

    // Close file
    fclose(file);

    // Allocate device memory
    double *d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeof(double) * n);
    cudaMalloc((void**)&d_output, sizeof(double));

    // Copy input array from host to device
    cudaMemcpy(d_input, h_input, sizeof(double) * n, cudaMemcpyHostToDevice);

    // Call kernel with 1 block
    parallelArraySum<<<1, NUM_THREADS>>>(d_input, d_output, n);

    // Copy the result back from device to host
    double h_output;
    cudaMemcpy(&h_output, d_output, sizeof(double), cudaMemcpyDeviceToHost);

    // Print the sum
    printf("%lf\n", h_output);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);

    return 0;
}