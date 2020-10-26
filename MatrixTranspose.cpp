/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>

// hip header file
#include "hip/hip_runtime.h"


#define WIDTH 32


#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 1
#define THREADS_PER_BLOCK_Y 1
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const int width) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    out[y * width + x] = in[x * width + y];
}

__global__ void matrixMult(float* out, float* A, const int wA, float* B, const int wB) {

	        float sum = 0.0;
	        int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
		    int j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

			// change to height
			if(i >= wA || j>= wB) return;
		    for (int k = 0; k < wA; k++) {
		        float a = A[i * wA + k];
		        float b = B[k * wB + j];
		        sum += a * b;
		    }
	        out[i*wA + j] = sum;
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

void matrixMultCPUReference(float* output, float* A, const unsigned int hA, const unsigned int wA, float* B, const unsigned int wB) {
    int i, j, k;
	float a, b;
	float c;

	for (i = 0; i < hA; i++) {
		for (j = 0; j < wB; j++) {
			c = 0;
			for (k = 0; k < wA; k++) {
				a = A[i * wA + k];
				b = B[k * wB + j];
				c += a * b;
		}
		output[i * wB + j] = c;
    }							    }
}

void printMatrix(float * res, int height, int width) {
	    int i;
	    printf("%d, %d\n", height, width);
	    for (i = 0; i < height*width; i++) {
	        printf("%f\n", res[i]);
	    }
}

int main() {
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrixA;
    float* gpuMatrixB;
    float* gpuTransposeMatrixOut;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    int i;
    int errors;

    Matrix = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix = (float*)malloc(NUM * sizeof(float));
    cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatrixA, NUM * sizeof(float));
    hipMalloc((void**)&gpuMatrixB, NUM * sizeof(float));
    hipMalloc((void**)&gpuTransposeMatrixOut, NUM * sizeof(float));

    // Memory transfer from host to device
    hipMemcpy(gpuMatrixA, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(gpuMatrixB, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);

	//printMatrix(Matrix, WIDTH, WIDTH);

    // Lauching kernel from host
    hipLaunchKernelGGL(matrixMult, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrixOut,
                    gpuMatrixA, WIDTH, gpuMatrixB, WIDTH);

    // Memory transfer from device to host
    hipMemcpy(TransposeMatrix, gpuTransposeMatrixOut, NUM * sizeof(float), hipMemcpyDeviceToHost);

    // CPU MatrixTranspose computation
    matrixMultCPUReference(cpuTransposeMatrix, Matrix, WIDTH, WIDTH, Matrix, WIDTH);

	//printMatrix(TransposeMatrix, WIDTH, WIDTH);
	printMatrix(cpuTransposeMatrix, WIDTH, WIDTH);

    // verify the results
    errors = 0;
    double eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
        if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
			//printf("Error: %f\n", TransposeMatrix[i]-cpuTransposeMatrix[i]);
            errors++;
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }

    // free the resources on device side
    hipFree(gpuMatrixA);
    hipFree(gpuMatrixB);
    hipFree(gpuTransposeMatrixOut);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

    return errors;
}
