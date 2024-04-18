#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TILE_SIZE 16

typedef long long ll;

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <filesystem>
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
using namespace std;

// Kernel function for 3D convolution
__global__ void outputTileConvolutionKernel(const unsigned char *inputImages, unsigned char *outputImages, const float *mask, int width, int height, int channels, int maskSize, int batchSize) {
    extern __shared__ float tile[];

    int origBlockDimX = blockDim.x - (maskSize - 1);
    int origBlockDimY = blockDim.y - (maskSize - 1);

    int maskRadius = maskSize / 2;

    int tileWidth = origBlockDimX + 2 * maskRadius;
    int tileHeight = origBlockDimY + 2 * maskRadius;

    int tileStartX = (blockIdx.x * origBlockDimX) - maskRadius;
    int tileStartY = (blockIdx.y * origBlockDimY) - maskRadius;

    int xIndex = blockIdx.x * origBlockDimX + threadIdx.x;
    int yIndex = blockIdx.y * origBlockDimY + threadIdx.y;
    int zIndex = blockIdx.z * blockDim.z + threadIdx.z;

    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < tileHeight; i++) {
            for (int j = 0; j < tileWidth; j++) {
                int x = tileStartX + j;
                int y = tileStartY + i;
                int tileIndex = (i * tileWidth + j) * channels + c;
                int imgIndex = (zIndex * height * width + y * width + x) * channels + c;
                bool validPixel = (x >= 0 && x < width && y >= 0 && y < height);
                tile[tileIndex] = validPixel ? inputImages[imgIndex] : 0;
            }
        }
    }

    __syncthreads();

    if (xIndex < width && yIndex < height && zIndex < batchSize && threadIdx.x < origBlockDimX && threadIdx.y < origBlockDimY) {
        float sum = 0.0;
        for (int i = 0; i < maskSize; i++)
            for (int j = 0; j < maskSize; j++)
                for (int c = 0; c < channels; c++)
                    sum += mask[i * maskSize + j] * tile[((threadIdx.y + i) * tileWidth + threadIdx.x + j) * channels + c];
        outputImages[zIndex * height * width + yIndex * width + xIndex] = (unsigned char)sum;
    }
}

int main(int argc, char **argv) {
    // Check for the correct number of arguments
    if (argc != 5) {
        printf("Usage: %s input_folder output_folder batch_size mask_file\n", argv[0]);
        return 1;
    }

    // Read command line arguments
    string inputFolder = argv[1];
    string outputFolder = argv[2];
    int batchSize = atoi(argv[3]);
    char* maskFile = argv[4];

    // Print the command line arguments
    printf("Input folder: %s\n", inputFolder.c_str());
    printf("Output folder: %s\n", outputFolder.c_str());
    printf("Batch size: %d\n", batchSize);
    printf("Mask file: %s\n", maskFile);
    printf("---------------------------------------------\n");

    // Check if the input folder exists
    if (!filesystem::exists(inputFolder)) {
        printf("Error: Input folder does not exist\n");
        return 1;
    }

    // Check if the mask file exists
    if (!filesystem::exists(maskFile)) {
        printf("Error: Mask file does not exist\n");
        return 1;
    }

    // Find the number of images in the input folder
    int numImages = 0;
    for (const auto &entry: filesystem::directory_iterator(inputFolder)) numImages++;
    printf("Number of images: %d\n", numImages);

    // Check if the number of images is less than the batch size
    if (numImages < batchSize) {
        printf("Error: Number of images is less than the batch size\n");
        return 1;
    }

    // Get the width, height, and number of channels of the first image in the input folder
    int width, height, channels;
    string firstImage = filesystem::directory_iterator(inputFolder)->path().string();
    unsigned char *image = stbi_load(firstImage.c_str(), &width, &height, &channels, STBI_rgb);
    if (image == NULL) {
        printf("Error: Failed to load image\n");
        return 1;
    }

    // Print the width, height, and number of channels of the image
    printf("Image width: %d\n", width);
    printf("Image height: %d\n", height);
    printf("Number of channels: %d\n", channels);

    // Calculate the size of the images
    int imgSize = width * height * channels;
    printf("Image size: %d\n", imgSize);
    printf("---------------------------------------------\n");

    // Allocate memory for input and output images
    unsigned char *inputImages = (unsigned char *)malloc(imgSize * batchSize * sizeof(unsigned char));
    unsigned char *outputImages = (unsigned char *)malloc(imgSize * batchSize * sizeof(unsigned char));

    // Read input images from the input folder and copy them to the inputImages array
    int i = 0;
    for (const auto &entry: filesystem::directory_iterator(inputFolder)) {
        string imagePath = entry.path().string();
        unsigned char *image = stbi_load(imagePath.c_str(), &width, &height, &channels, STBI_rgb);
        if (image == NULL) {
            printf("Error: Failed to load image\n");
            return 1;
        }
        memcpy(inputImages + i * imgSize, image, imgSize * sizeof(unsigned char));
        stbi_image_free(image);
        i++;
        if (i == batchSize) break;
    }
    
    // Read convolution mask from the provided text file
    FILE *fp = fopen(maskFile, "r");
    if (fp == NULL) {
        printf("Error: Unable to open mask file\n");
        return 1;
    }
    int maskSize;
    fscanf(fp, "%d", &maskSize);
    printf("Mask size: %d\n", maskSize);
    float *mask = (float *)malloc(maskSize * maskSize * sizeof(float));
    if (mask == NULL) {
        printf("Error: Memory allocation failed for mask\n");
        fclose(fp);
        return 1;
    }
    for (int i = 0; i < maskSize * maskSize; i++) {
        fscanf(fp, "%f", &mask[i]);
    }
    fclose(fp);

    // Print the mask
    printf("Mask:\n");
    for (int i = 0; i < maskSize; i++) {
        for (int j = 0; j < maskSize; j++) {
            printf("%f ", mask[i * maskSize + j]);
        }
        printf("\n");
    }
    printf("---------------------------------------------\n");

    // Allocate memory on GPU for input images, output images, and mask
    unsigned char *d_inputImages, *d_outputImages;
    float *d_mask;
    cudaMalloc(&d_inputImages, imgSize * batchSize * sizeof(unsigned char));
    cudaMalloc(&d_outputImages, imgSize * batchSize * sizeof(unsigned char));
    cudaMalloc(&d_mask, maskSize * maskSize * sizeof(float));

    // Copy input images and mask from host to device
    cudaMemcpy(d_mask, mask, maskSize * maskSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputImages, inputImages, imgSize * batchSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    dim3 block(TILE_SIZE + maskSize - 1, TILE_SIZE + maskSize - 1, 1);
    dim3 grid((width + block.x - 1) / TILE_SIZE, (height + block.y - 1) / TILE_SIZE, (batchSize + block.z - 1) / block.z);

    // Calculate shared memory size
    int sharedMemorySize = pow(TILE_SIZE + maskSize - 1, 2) * channels * sizeof(float);

    // Launch kernel for all images in the batch
    outputTileConvolutionKernel<<<grid, block, sharedMemorySize>>>(d_inputImages, d_outputImages, d_mask, width, height, channels, maskSize, batchSize);

    // Copy results (output images) from device to host
    cudaMemcpy(outputImages, d_outputImages, imgSize * batchSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Create output directory if it does not exist
    if (!filesystem::exists(outputFolder)) {
        filesystem::create_directory(outputFolder);
    }

    // Save output images to the specified output folder
    for (int i = 0; i < batchSize; i++) {
        string outputImagePath = outputFolder + "/" + to_string(i) + ".jpg";
        stbi_write_jpg(outputImagePath.c_str(), width, height, 1, outputImages + i * width * height, width);
    }
    printf("Output images saved to %s\n", outputFolder.c_str());

    // Free memory
    free(inputImages);
    free(outputImages);
    free(mask);
    cudaFree(d_inputImages);
    cudaFree(d_outputImages);
    cudaFree(d_mask);

    return 0;
}
