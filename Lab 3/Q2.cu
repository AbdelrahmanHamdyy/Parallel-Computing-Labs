#include <stdio.h>
#include <stdlib.h>

// Define the block size to be used in the kernel
#define NUM_THREADS 1024

// Define a type for long long
typedef long long ll;

// Standard binary search function which runs on the device
__device__ ll binarySearch(double *input, ll left, ll right, double target) {
    while (left <= right) {
        ll mid = (left + right) / 2;

        if (input[mid] == target) {
            return mid;
        } else if (input[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

__global__ void parallelBinarySearch(double *input, ll n, double target, ll *output) {
    // Shared memory to store the result of each thread
    __shared__ ll result[NUM_THREADS];

    // Thread ID
    ll tid = threadIdx.x;

    // Calculate the chunk size
    ll chunkSize = (n + NUM_THREADS - 1) / NUM_THREADS;

    // Calculate the start and end index of the chunk for this thread
    ll left = tid * chunkSize;
    ll right = min(left + chunkSize - 1, n - 1);

    // Perform binary search on the chunk of the array
    result[tid] = binarySearch(input, left, right, target);

    // Wait for all threads to finish
    __syncthreads();

    // Perform parallel reduction to find the first non-negative result
    for (ll stride = NUM_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride && result[tid] == -1)
            result[tid] = result[tid + stride];

        // Synchronize threads
        __syncthreads();
    }

    // Set the result of the block in the output
    if (tid == 0)
        *output = result[0];
}

int main(int argc, char *argv[]) {
    // Read the target value from the command line
    double target = atof(argv[2]);

    // Open the file for reading
    FILE *file = fopen(argv[1], "r");

    // Check if the file was opened successfully
    if (!file) {
        fprintf(stderr, "Error opening file.\n");
        exit(1);
    }

    // Read the number of elements from the file
    ll n = 0;
    double val; // Temp variable to read the elements
    while (fscanf(file, "%lf", &val) == 1)
        n++; // Increment the count of elements in the array
    
    // Reset file pointer to beginning
    fseek(file, 0, SEEK_SET);

    // Allocate host memory for the array
    double *h_input = (double*)malloc(sizeof(double) * n);

    // Read array elements from the file
    for (ll i = 0; i < n; i++)
        fscanf(file, "%lf", &h_input[i]);

    // Close the file
    fclose(file);

    // Allocate device memory
    double *d_input;
    ll *d_result;
    cudaMalloc((void**)&d_input, sizeof(double) * n);
    cudaMalloc((void**)&d_result, sizeof(ll));

    // Copy input array from host to device
    cudaMemcpy(d_input, h_input, sizeof(double) * n, cudaMemcpyHostToDevice);

    // Call kernel with one block
    parallelBinarySearch<<<1, NUM_THREADS>>>(d_input, n, target, d_result);

    // Copy the result back from device to host
    ll h_result;
    cudaMemcpy(&h_result, d_result, sizeof(ll), cudaMemcpyDeviceToHost);

    // Print the result
    printf("%lld\n", h_result);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_result);

    // Free host memory
    free(h_input);

    return 0;
}