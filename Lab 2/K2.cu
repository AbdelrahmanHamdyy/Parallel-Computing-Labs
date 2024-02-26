#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define MAX_ERR 1e-6

__global__ void matrix_add(float **out, float **a, float **b, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // A single thread loops over all columns and calculates the sum of column elements in that row only
        for (int j = 0; j < cols; j++) {
            out[row][j] = a[row][j] + b[row][j];
        }
    }
}

int main(int argc, char *argv[]) {
    // Set up input and output files
    FILE *inputFile = fopen(argv[1], "r");
    FILE *outputFile = fopen(argv[2], "w");

    // Check if files are opened correctly
    if (!inputFile || !outputFile) {
        fprintf(stderr, "Error opening file\n");
        exit(1);
    }

    int testCases; // Number of test cases
    fscanf(inputFile, "%d", &testCases);
    
    for (int t = 0; t < testCases; t++) {
        // Read the number of rows and columns of the matrices
        int rows, cols;
        fscanf(inputFile, "%d %d", &rows, &cols);

        // Define host and device matrices
        float **a, **b, **out;
        float **d_a, **d_b, **d_out;

        // Allocate host memory for matrices
        a = (float**)malloc(sizeof(float*) * rows);
        b = (float**)malloc(sizeof(float*) * rows);
        out = (float**)malloc(sizeof(float*) * rows);
        
        // Allocate memory for each row in the arrays
        for (int i = 0; i < rows; i++) {
            a[i] = (float*)malloc(sizeof(float) * cols);
            b[i] = (float*)malloc(sizeof(float) * cols);
            out[i] = (float*)malloc(sizeof(float) * cols);
        }

        // Read the first matrix data from file
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fscanf(inputFile, "%f", &a[i][j]);
            }
        }

        // Read the second matrix data from file
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fscanf(inputFile, "%f", &b[i][j]);
            }
        }

        // Allocate device memory for matrices
        cudaMalloc((void***)&d_a, sizeof(float*) * rows);
        cudaMalloc((void***)&d_b, sizeof(float*) * rows);
        cudaMalloc((void***)&d_out, sizeof(float*) * rows);

        // These will be used to allocate arrays of size = cols in device memory for each row
        // Since we can't allocate 2D arrays directly in device memory
        float **d_a_data = (float**)malloc(sizeof(float*) * rows);
        float **d_b_data = (float**)malloc(sizeof(float*) * rows);
        float **d_out_data = (float**)malloc(sizeof(float*) * rows);

        // Allocate places for all matrix elements
        for (int i = 0; i < rows; i++) {
            cudaMalloc((void**)&d_a_data[i], sizeof(float) * cols);
            cudaMalloc((void**)&d_b_data[i], sizeof(float) * cols);
            cudaMalloc((void**)&d_out_data[i], sizeof(float) * cols);
        }

        // Transfer pointers to device memory
        cudaMemcpy(d_a, d_a_data, sizeof(float*) * rows, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, d_b_data, sizeof(float*) * rows, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, d_out_data, sizeof(float*) * rows, cudaMemcpyHostToDevice);

        // Transfer data from host to device memory
        for (int i = 0; i < rows; i++) {
            cudaMemcpy(d_a_data[i], a[i], sizeof(float) * cols, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b_data[i], b[i], sizeof(float) * cols, cudaMemcpyHostToDevice);
        }

        // Define block size and grid size
        dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

        // Execute kernel
        matrix_add<<<gridSize, blockSize>>>(d_out, d_a, d_b, rows, cols);

        // Transfer data back to host memory
        for (int i = 0; i < rows; i++) {
            cudaMemcpy(out[i], d_out_data[i], sizeof(float) * cols, cudaMemcpyDeviceToHost);
        }

        // Assertion
        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
            assert(fabs(out[i][j] - a[i][j] - b[i][j]) < MAX_ERR);
          }
        }

        // Write result to output file
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(outputFile, "%f ", out[i][j]);
            }
            fprintf(outputFile, "\n");
        }

        // Free device memory
        for (int i = 0; i < rows; i++) {
            cudaFree(d_a_data[i]);
            cudaFree(d_b_data[i]);
            cudaFree(d_out_data[i]);
        }
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);

        // Free temp data
        free(d_a_data);
        free(d_b_data);
        free(d_out_data);

        // Free host memory
        for (int i = 0; i < rows; i++) {
            free(a[i]);
            free(b[i]);
            free(out[i]);
        }
        free(a);
        free(b);
        free(out);
    }

    fclose(inputFile);
    fclose(outputFile);

    printf("PASSED\n");

    return 0;
}