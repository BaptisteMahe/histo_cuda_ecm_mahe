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
#define THREADS_PER_BLOCK 1024
#define FIST_RELEVANT_ASCII 32
#define LAST_RELEVANT_ASCII 126
#define FIRST_UPP_ASCII 65
#define LAST_UPP_ASCII 90

////////////////////////////////////////////////////////////////////////////////
// Declarations
void generateHisto(char* inputFileName, char* outputFileName);

void writeOutputCSV(int result[NB_ASCII], char* outputFileName);

void processBatchInKernel(  char h_data[MAX_LINE][MAX_CHAR],
                            int nbLine,
                            int lineSize,
                            int resultSize,
                            int resultStorage[NB_ASCII]);

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

    __global__ int s_result[NB_ASCII];
    
    if (ti < nbLine) {
		char* line = (char *)((char*)d_data + ti * pitch);
		int index = 0;
		int currentLetter = line[index];

		while (currentLetter > 0) {
	    	atomicAdd(&s_result[currentLetter], 1);
	    	index++;
	    	currentLetter = line[index];
		}

        __syncthreads();

        if (ti == 0) {
            for (int i = 0; i < NB_ASCII; i++) {
                d_result[i] = s_result[i];
            }
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

	printf("\n%s Starting...\n\n", argv[0]);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

	generateHisto(inputFileName, outputFileName);

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    printf("Input file processed successfully.\n");
    if (outputFileName) {
        printf("Check results in %s.\n\n", outputFileName);
    } else {
        printf("Check results in out.csv.\n\n");
    }

	exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Generate the Histogram
//! @param inputFileName input name of the file to process
//! @param outputFileName input name of the file to output the histogram in
////////////////////////////////////////////////////////////////////////////////
void generateHisto(char* inputFileName, char* outputFileName) {
    
    // Compute result and data sizes
    unsigned int resultSize = NB_ASCII * sizeof(int);
    unsigned int lineSize = MAX_CHAR * sizeof(char);

    // Load input file
    FILE *inputFile = NULL;
    inputFile = fopen(inputFileName, "r");

    if (!inputFile) {
        printf("Wrong input file\n");
        printHelper();
		exit(EXIT_FAILURE);
    }

    // Allocate host memory
    char str[MAX_CHAR];
    char h_data[MAX_LINE][MAX_CHAR];
    int resultStorage[NB_ASCII];
    int nbLine = 0;
    int batchNum = 1;
    
    while (fgets(str, MAX_CHAR, inputFile)) {
	
		if (nbLine == MAX_LINE) {

            printf("Batch N°%i: %i lines. \n", batchNum, nbLine);

	    	processBatchInKernel(h_data, nbLine, lineSize, resultSize, resultStorage);
            
            nbLine = 0;
            batchNum++;
		}

        strcpy(h_data[nbLine], str);
        nbLine++;
    }
    
    printf("Batch N°%i: %i lines. \n", batchNum, nbLine);

    processBatchInKernel(h_data, nbLine, lineSize, resultSize, resultStorage);
    
    fclose(inputFile);
    
    //write the output
    writeOutputCSV(resultStorage, outputFileName);
}

////////////////////////////////////////////////////////////////////////////////
//! Send batch data to kernel and store the output in resultStorage
//! @param d_data input pointer to the allocated memory for the input data on the device
//! @param h_data input the list of strings to process
//! @param nbLine input number of lines to process for the current batch
//! @param pitch input pitch size of the array in the device 
//! @param lineSize input size of a single line
//! @param d_result input pointer to the allo
//! @param resultSize input pointer to the allocated memory for the output data on the device
//! @param resultStorage output result of the computation as an array
////////////////////////////////////////////////////////////////////////////////

void processBatchInKernel(  char h_data[MAX_LINE][MAX_CHAR],
                            int nbLine,
                            int lineSize,
                            int resultSize,
                            int resultStorage[NB_ASCII]) {

    // Allocate device memory
    int* d_result;
    char* d_data;
    size_t pitch;
    checkCudaErrors(cudaMallocPitch((void **) &d_data, &pitch, lineSize, MAX_LINE));
    checkCudaErrors(cudaMalloc((void **) &d_result, resultSize));

    // Allocate memory for result in host
    int h_result[NB_ASCII];

    // Setup execution parameters
    dim3  grid((nbLine + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
    dim3  threads(THREADS_PER_BLOCK, 1, 1);

    // Copy data to device
    checkCudaErrors(cudaMemcpy2D(d_data, pitch, h_data, lineSize, lineSize, MAX_LINE, cudaMemcpyHostToDevice));
    
    // Execute the kernel
    kernelFunction<<< grid, threads, 0 >>>(d_data, d_result, nbLine, pitch);
    getLastCudaError("Kernel execution failed");
    
    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(&h_result, d_result, resultSize, cudaMemcpyDeviceToHost));

    for (int index = 0; index < NB_ASCII; index++) {
        resultStorage[index] += h_result[index];
    }

    // Cleanup memory
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_result));
}

////////////////////////////////////////////////////////////////////////////////
//! Write the given output to the CSV file
//! @param result input the given ouput of the computations as an array of int
//! @param outputFileName input file name to write in
////////////////////////////////////////////////////////////////////////////////

void writeOutputCSV(int result[NB_ASCII], char* outputFileName) {
	FILE *outputFile;
	char asciiChar;

    if (outputFileName) {
        outputFile = fopen(outputFileName, "w+");
    } else {
        outputFile = fopen("out.csv", "w+");
    }
	
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