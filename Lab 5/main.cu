// Includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Define the cucheck_dev macro for error checking in CUDA
#define cucheck_dev(call)                                   \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
    assert(0);                                              \
  }                                                         \
}

// G_UIDs
__device__ int g_uids = 0;

// Define the block size to be used in the kernel
#define NUM_THREADS 32

// Define the recursion limit
#define MAX_DEPTH 4

// Define a type for long long
typedef long long ll;

// Kernel function for dynamic parallelism
__global__ void cdp_kernel(int max_depth, int depth, int thread, int parent_uid, int* output) {
    // We create a unique ID per block. Thread 0 does that and shares the value with the other threads.
    __shared__ int s_uid;
    if (threadIdx.x == 0)
    {
      s_uid = atomicAdd(&g_uids, 1);
      output[0] = s_uid;
      printf("BLOCK %d launched by thread %d of block %d\n", s_uid, thread, parent_uid);
    }

    // Wait for all threads to finish
    __syncthreads();

    // We launch new blocks if we haven't reached the max_depth yet.
    if (depth >= max_depth)
      return;

    cdp_kernel<<<gridDim.x, blockDim.x>>>(max_depth, depth+1, threadIdx.x, s_uid, output);
    cucheck_dev(cudaGetLastError());
    __syncthreads();
}

// Standard binary search function which runs on the device
__device__ void binarySearch(double *input, ll left, ll right, double target, ll *output) {
    while (left <= right) {
        ll mid = (left + right) / 2;

        if (input[mid] == target) {
            *output = mid;
            return;
        } else if (input[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
}

// Parallel Binary Search using multiple threads without dynamic parallelism
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
    binarySearch(input, left, right, target, &result[tid]);

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

// [CDP] Recursive Binary Search using Dynamic Parallelism & a single thread
__global__ void singleThreadRecursiveBinarySearch(double *input, ll left, ll right, double target, ll *output, int depth, int maxDepth) {
    // Goal: Implement Binary Search using dynamic parallelism in CUDA (CDP)
    //////////// STEPS ////////////
    // 1. Check if the left index is less than or equal to the right index
    // 2. Calculate the mid index
    // 3. Check if the mid element is equal to the target element
    // 4. If yes, set the output to the mid index and return
    // 5. If no, check if the mid element is less than the target element
    // 6. If yes, call the kernel recursively with the mid + 1 index as the left index
    // 7. If no, call the kernel recursively with the mid - 1 index as the right index
    // 8. If the left index is greater than the right index, set the output to -1 and return

    // Check the depth
    if (depth >= maxDepth) {
        // If the maximum depth is reached
        // Perform sequential binary search
        binarySearch(input, left, right, target, output);
        return;
    }

    // Check if the left index is less than or equal to the right index
    if (left <= right) {
        // Calculate the mid index
        ll mid = left + (right - left) / 2;

        // Check if the mid element is equal to the target element
        if (input[mid] == target) {
            // If yes, set the output to the mid index and return
            *output = mid;
            return;
        } else if (input[mid] < target) { // If no, check if the mid element is less than the target element
            // If yes, call the kernel recursively with the mid + 1 index as the left index
            singleThreadRecursiveBinarySearch<<<1, 1>>>(input, mid + 1, right, target, output, depth + 1, maxDepth);
        } else {
            // If no, call the kernel recursively with the mid - 1 index as the right index
            singleThreadRecursiveBinarySearch<<<1, 1>>>(input, left, mid - 1, target, output, depth + 1, maxDepth);
        }
    } else {
        // If the left index is greater than the right index, set the output to -1 and return
        *output = -1;
        return;
    }
}

// [CDP] Recursive Binary Search using Dynamic Parallelism & multiple threads
__global__ void parallelRecursiveBinarySearch(double *input, ll left, ll right, double target, ll *output, int depth, int maxDepth) {
    // The goal of this kernel is to implement a parallel version of the recursive binary search algorithm
    // The kernel should use dynamic parallelism to call itself recursively
    // We will divide the array into chunks and each thread will be responsible for a chunk
    // Each thread will call the kernel recursively on its chunk
    // If the maximum recursion depth is reached, the thread will continue with sequential binary search on its chunk

    // [Size] Calculate the capacity of the array
    ll numberOfElements = right - left + 1;

    // [Thread Index] Thread ID within the block
    ll threadIndex = threadIdx.x;

    // [Split Flag] Determine if the array should be split into chunks or not
    bool split = numberOfElements > NUM_THREADS;

    // [Last Thread Flag] Determine if this thread is the last one
    bool lastThread = threadIndex == NUM_THREADS - 1;

    // [First Thread Flag] Determine if this thread is the first one placed at the root meaning that it still didn't traverse the tree
    bool firstThread = threadIndex == 0 && depth == 0;

    // [Chunk Size] Calculate the chunk size for each thread
    ll chunkSize = numberOfElements / NUM_THREADS;

    // [Initialization] We need to set the output to -1 for the first thread before any recursive calls
    if (firstThread) *output = -1;

    // [Depth] Check if the depth is less than the maximum depth and the array can be split even furthur
    if (depth < maxDepth && split) {
        /* UPDATE Left & Right */
        // Calculate the start index according to the thread index
        // So that each thread will work on a separate chunk
        left += threadIndex * chunkSize;

        // Re-calculate the new end index to the current chunk
        // If this is the last thread, the end index will be the last element in the array to avoid out-of-bounds access
        // If not, then it will be starting from the current left index plus the chunk size minus 1 (Spanning the chunk size)
        right = lastThread ? right : left + chunkSize - 1;

        // [CDP] Recursive call to the kernel with the new chunk and incremented depth
        parallelRecursiveBinarySearch<<<1, NUM_THREADS>>>(input, left, right, target, output, depth + 1, maxDepth);
    } else { // Maximum depth reached or no splits left to apply on the input
        // [Binary Search] If yes, perform sequential binary search on the current chunk
        binarySearch(input, left, right, target, output);
    }
}

int main(int argc, char *argv[]) {
    // Check if the number of arguments is correct
    if (argc != 3) {
        fprintf(stderr, "Usage: %s inputFile target\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Read the target value from the command line
    double target = atof(argv[2]);

    // Open the file for reading
    FILE *file = fopen(argv[1], "r");

    // Check if the file was opened successfully
    if (!file) {
        fprintf(stderr, "Error opening file.\n");
        exit(EXIT_FAILURE);
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

    // Set the recursion limit
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    // Call kernel with one block
    parallelRecursiveBinarySearch<<<1, NUM_THREADS>>>(d_input, 0, n - 1, target, d_result, 0, MAX_DEPTH);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

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

    // Success!
    return EXIT_SUCCESS;
}