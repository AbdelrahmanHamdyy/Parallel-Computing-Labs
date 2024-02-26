#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define MAX_ERR 1e-6

__global__ void matrix_vector_mult(float *out, float **matrix, float *vector, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += matrix[idx][j] * vector[j];
        }
        out[idx] = sum;
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

    // Read the number of test cases
    int testCases;
    fscanf(inputFile, "%d", &testCases);

    for (int t = 0; t < testCases; t++) {
        // Read the number of rows and columns of the matrix
        int rows, cols;
        fscanf(inputFile, "%d %d", &rows, &cols);

        // Define host and device matrix & vector to multiply, and the output vector as well
        float **matrix;
        float *vector, *out;
        float **d_matrix, *d_vector, *d_out;

        // Allocate host memory for matrix, vector, and output
        matrix = (float**)malloc(sizeof(float*) * rows);
        for (int i = 0; i < rows; i++) {
            matrix[i] = (float*)malloc(sizeof(float) * cols);
        }

        vector = (float*)malloc(sizeof(float) * rows);
        out = (float*)malloc(sizeof(float) * rows);

        // Read matrix data from file
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fscanf(input_file, "%f", &matrix[i][j]);
            }
        }

        // Read vector data from file
        for (int i = 0; i < rows; i++) {
            fscanf(input_file, "%f", &vector[i]);
        }

        // Allocate device memory for matrix, vector, and output
        cudaMalloc((void***)&d_matrix, sizeof(float*) * rows);
        cudaMalloc((void**)&d_vector, sizeof(float) * rows);
        cudaMalloc((void**)&d_out, sizeof(float) * rows);

        float **d_matrix_data = (float**)malloc(sizeof(float*) * rows);
        for (int i = 0; i < rows; i++) {
            cudaMalloc((void**)&d_matrix_data[i], sizeof(float) * cols);
        }

        cudaMemcpy(d_vector, vector, sizeof(float) * rows, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, out, sizeof(float) * rows, cudaMemcpyHostToDevice);

        // Transfer data from host to device memory
        for (int i = 0; i < rows; i++) {
            cudaMemcpy(d_matrix_data[i], matrix[i], sizeof(float) * cols, cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_matrix, d_matrix_data, sizeof(float*) * rows, cudaMemcpyHostToDevice);

        // Define block size and grid size
        int blockSize = 256;
        int gridSize = (rows + blockSize - 1) / blockSize;

        // Execute kernel
        matrix_vector_mult<<<gridSize, blockSize>>>(d_out, d_matrix, d_vector, rows, cols);

        // Transfer data back to host memory
        cudaMemcpy(out, d_out, sizeof(float) * rows, cudaMemcpyDeviceToHost);

        // Write result to output file
        for (int i = 0; i < rows; i++) {
            fprintf(output_file, "%f\n", out[i]);
        }

        // Free device memory
        for (int i = 0; i < rows; i++) {
            cudaFree(d_matrix_data[i]);
        }
        cudaFree(d_matrix);
        cudaFree(d_vector);
        cudaFree(d_out);
        free(d_matrix_data);

        // Free host memory
        for (int i = 0; i < rows; i++) {
            free(matrix[i]);
        }
        free(matrix);
        free(vector);
        free(out);
    }

    fclose(inputFile);
    fclose(outputFile);

    printf("PASSED\n");

    return 0;
}