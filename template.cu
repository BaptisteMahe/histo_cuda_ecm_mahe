////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include "common.h"

// define constants
#define MAX_LINE 200000
#define MAX_CHAR 40
#define NB_ASCII 128

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void generateHisto(char* inputFileName, char* outputFileName);

void writeOutputCSV(int result[NB_ASCII], char* outputFileName);

void startTimer(StopWatchInterface* timer);

void stopTimer(StopWatchInterface* timer);

////////////////////////////////////////////////////////////////////////////////
//! Kernel function to execute the computation in threads
//! @param d_data  input data in global memory
//! @param d_result  output result as array in global memory
//! @param nbLine  input size of the data in global memory
//! @param pitch  input pitch size of in the data global memory
////////////////////////////////////////////////////////////////////////////////

__global__ 
void kernelFunction(char* d_data, int* d_result, int nbLine, size_t pitch) {
    
    const unsigned int tidb = threadIdx.x;
    const unsigned int ti = blockIdx.x*blockDim.x + tidb;
    
    if (ti < nbLine) {
		char* line = (char *)((char*)d_data + ti * pitch);
		int index = 0;
		int currentLetter = line[index];

		while (currentLetter > 0) {
	    	atomicAdd(&d_result[currentLetter], 1);
	    	index++;
	    	currentLetter = line[index];
		}
    } 
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
	
	int c;
	char *inputFileName = NULL;
	char *outputFileName = NULL;

	while ((c = getopt (argc, argv, "i:o:")) != -1)
		switch(c) {
			case 'i':
				inputFileName = optarg;
				break;
			case 'o':
				outputFileName = optarg;
				break;
			default:
				break;
		}

	printf("%s Starting...\n\n", argv[0]);

	generateHisto(inputFileName, outputFileName);

	exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Generate the Histogram
////////////////////////////////////////////////////////////////////////////////
void generateHisto(char* inputFileName, char* outputFileName) {

    StopWatchInterface *timer = 0;
    startTimer(timer);

    // Print MemInfo
    size_t memfree, memtotal;
    checkCudaErrors(cudaMemGetInfo(&memfree, &memtotal));

    unsigned int threadsPerBlock = 1024;
    unsigned int resultSize = NB_ASCII * sizeof(int);
    unsigned int lineSize = MAX_CHAR * sizeof(char);


    // Allocate device memory
    int* d_result;
    char* d_data;
    size_t pitch;
    checkCudaErrors(cudaMallocPitch((void **) &d_data, &pitch, lineSize, MAX_LINE));
    checkCudaErrors(cudaMalloc((void **) &d_result, resultSize));

    // Setup execution parameters
    dim3  grid((MAX_LINE + threadsPerBlock - 1) / threadsPerBlock, 1, 1);
    dim3  threads(threadsPerBlock, 1, 1);

    // Load input file data
    FILE *inputFile = NULL;
    inputFile = fopen(inputFileName, "r");
    if (!inputFile) {
        printf("Wrong input file\n");
		exit(EXIT_FAILURE);
    }

    // Allocate host memory
    char str[MAX_CHAR];
    char h_data[MAX_LINE][MAX_CHAR];
    int h_result[NB_ASCII];
    int totalResult[NB_ASCII];
    int nbLine = 0;
    
    while (fgets(str, MAX_CHAR, inputFile)) {
	
		if (nbLine == MAX_LINE) {

            printf("Loaded %i lines \n", nbLine);

	    	// Copy data to device
            checkCudaErrors(cudaMemcpy2D(d_data, pitch, h_data, lineSize, lineSize, MAX_LINE, cudaMemcpyHostToDevice));
            // Execute the kernel
            kernelFunction<<< grid, threads, 0 >>>(d_data, d_result, nbLine, pitch);
            getLastCudaError("Kernel execution failed");
            // Copy result from device to host
            checkCudaErrors(cudaMemcpy(&h_result, d_result, resultSize, cudaMemcpyDeviceToHost));

            for (int index = 0; index < NB_ASCII; index++) {
                totalResult[index] += h_result[index];
            }
            
            nbLine = 0;
		}

        strcpy(h_data[nbLine], str);
        nbLine++;
    }
    
    printf("Loaded %i lines \n", nbLine);

    // Copy data to device
    checkCudaErrors(cudaMemcpy2D(d_data, pitch, h_data, lineSize, lineSize, MAX_LINE, cudaMemcpyHostToDevice));
    // Execute the kernel
    kernelFunction<<< grid, threads, 0 >>>(d_data, d_result, nbLine, pitch);
    getLastCudaError("Kernel execution failed");
    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(&h_result, d_result, resultSize, cudaMemcpyDeviceToHost));

    for (int index = 0; index < NB_ASCII; index++) {
        totalResult[index] += h_result[index];
    } 
    
    fclose(inputFile);
    
    //write the output
    writeOutputCSV(h_result, outputFileName);

    // cleanup memory
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_result));

    stopTimer(timer);
}

////////////////////////////////////////////////////////////////////////////////
//! Write the given output to the CSV file
////////////////////////////////////////////////////////////////////////////////

void writeOutputCSV(int result[NB_ASCII], char* outputFileName) {
	FILE *outputFile;
	char asciiChar;

	outputFile = fopen(outputFileName, "w+");
	
	for (int index = 32; index < 127; index++) {
		asciiChar = index;
		fprintf(outputFile, "%c: %i\n", asciiChar, result[index]);
	}

	fclose(outputFile);
}

////////////////////////////////////////////////////////////////////////////////
//! Start the timer
////////////////////////////////////////////////////////////////////////////////
void startTimer(StopWatchInterface* timer) {
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
}

////////////////////////////////////////////////////////////////////////////////
//! Stop the timer
////////////////////////////////////////////////////////////////////////////////
void stopTimer(StopWatchInterface* timer) {
  sdkStopTimer(&timer);
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);
}
