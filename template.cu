// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

// Includes, CUDA
#include <cuda_runtime.h>

// Includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include "common.h"

// Define constants
#define MAX_LINE 200000
#define MAX_CHAR 40
#define NB_ASCII 128
#define FIST_RELEVANT_ASCII 32
#define LAST_RELEVANT_ASCII 126
#define FIRST_UPP_ASCII 65
#define LAST_UPP_ASCII 90

////////////////////////////////////////////////////////////////////////////////
// Declarations
void generateHisto(char* inputFileName, char* outputFileName);

void writeOutputCSV(int result[NB_ASCII], char* outputFileName);

void processBatchInKernel(  char** d_data,
                            char h_data[MAX_LINE][MAX_CHAR],
                            int nbLine,
                            size_t pitch,
                            int lineSize,
                            int** d_result,
                            int resultSize,
                            int totalResult[NB_ASCII],
                            int threadsPerBlock);

void printHelper();
                            
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
	char *outputFileName = "out.csv";

	while ((c = getopt (argc, argv, "i:o:h")) != -1)
		switch(c) {
			case 'i':
				inputFileName = optarg;
				break;
			case 'o':
				outputFileName = optarg;
				break;
            case 'h':
                printHelper();
                exit(EXIT_SUCCESS);
			default:
				break;
		}

	printf("%s Starting...\n\n", argv[0]);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

	generateHisto(inputFileName, outputFileName);

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

	exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Generate the Histogram
////////////////////////////////////////////////////////////////////////////////
void generateHisto(char* inputFileName, char* outputFileName) {

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

    // Load input file
    FILE *inputFile = NULL;
    inputFile = fopen(inputFileName, "r");
    if (!inputFile) {
        printf("Wrong input file\n");
		exit(EXIT_FAILURE);
    }

    // Allocate host memory
    char str[MAX_CHAR];
    char h_data[MAX_LINE][MAX_CHAR];
    int totalResult[NB_ASCII];
    int nbLine = 0;
    int batchNum = 1;
    
    while (fgets(str, MAX_CHAR, inputFile)) {
	
		if (nbLine == MAX_LINE) {

            printf("Batch N°%i: %i lines. \n", batchNum, nbLine);

	    	processBatchInKernel(&d_data, h_data, nbLine, pitch, lineSize, &d_result, resultSize, totalResult, threadsPerBlock);
            
            nbLine = 0;
            batchNum++;
		}

        strcpy(h_data[nbLine], str);
        nbLine++;
    }
    
    printf("Batch N°%i: %i lines. \n", batchNum, nbLine);

    processBatchInKernel(&d_data, h_data, nbLine, pitch, lineSize, &d_result, resultSize, totalResult, threadsPerBlock);
    
    fclose(inputFile);
    
    //write the output
    writeOutputCSV(totalResult, outputFileName);

    // cleanup memory
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_result));
}

////////////////////////////////////////////////////////////////////////////////
//! Send batch data to kernel and store the output in totalResult
////////////////////////////////////////////////////////////////////////////////

void processBatchInKernel(  char** d_data,
                            char h_data[MAX_LINE][MAX_CHAR],
                            int nbLine,
                            size_t pitch,
                            int lineSize,
                            int** d_result,
                            int resultSize,
                            int totalResult[NB_ASCII],
                            int threadsPerBlock) {
    // Allocate memory for result in host
    int h_result[NB_ASCII];

    // Setup execution parameters
    dim3  grid((nbLine + threadsPerBlock - 1) / threadsPerBlock, 1, 1);
    dim3  threads(threadsPerBlock, 1, 1);

    // Copy data to device
    checkCudaErrors(cudaMemcpy2D(*d_data, pitch, h_data, lineSize, lineSize, MAX_LINE, cudaMemcpyHostToDevice));
    
    // Execute the kernel
    kernelFunction<<< grid, threads, 0 >>>(*d_data, *d_result, nbLine, pitch);
    getLastCudaError("Kernel execution failed");
    
    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(&h_result, *d_result, resultSize, cudaMemcpyDeviceToHost));

    for (int index = 0; index < NB_ASCII; index++) {
        totalResult[index] = h_result[index];
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Write the given output to the CSV file
////////////////////////////////////////////////////////////////////////////////

void writeOutputCSV(int result[NB_ASCII], char* outputFileName) {
	FILE *outputFile;
	char asciiChar;

	outputFile = fopen(outputFileName, "w+");
	
	for (int index = FIST_RELEVANT_ASCII; index <= LAST_RELEVANT_ASCII; index++) {

        if (index >= FIRST_UPP_ASCII && index <= LAST_UPP_ASCII) {
            // Add uppercase count to char count
            result[index + 32] += result[index];
        } else {
            // Print count in file
            asciiChar = index;
		    fprintf(outputFile, "%c: %i\n", asciiChar, result[index]);
        }

	}

	fclose(outputFile);
}

////////////////////////////////////////////////////////////////////////////////
//! Print information for user
////////////////////////////////////////////////////////////////////////////////

void printHelper() {
    printf("\n");
    printf("Usage :\n");
    printf("\t- -i <inputFileName>  (required)\n");
    printf("\t- -o <outputFileName> (default is 'out.csv')\n");
    printf("Info :\n");
    printf("\t- The input file should be a text file with a maximum of %i characters per line.\n", MAX_CHAR);
    printf("\t- The input file will be processed by batches of %i lines.\n", MAX_LINE);
    printf("\t- There is no limit regarding the number of lines of the input file.\n");
    printf("\n");
}