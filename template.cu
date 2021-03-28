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
////////////////////////////////////////////////////////////////////////////////
void generateHisto(char* inputFileName, char* outputFileName);

void writeOutputCSV(unsigned long int result[NB_ASCII], char* outputFileName);

void processBatchInKernel(  char** d_data,
                            char h_data[MAX_LINE][MAX_CHAR],
                            int nbLine,
                            size_t pitch,
                            int lineSize,
                            unsigned long int** d_result,
                            int resultSize,
                            unsigned long int resultStorage[NB_ASCII]);

void printHelper();

////////////////////////////////////////////////////////////////////////////////
//! Kernel function to execute the computation in threads using only Global Memory
//! @param d_data  input data in global memory
//! @param d_result  output result as array in global memory
//! @param nbLine  input size of the data in global memory
//! @param pitch  input pitch size of in the data global memory
////////////////////////////////////////////////////////////////////////////////

__global__ 
void kernelGlobalMem(char* d_data, unsigned long int* d_result, int nbLine, size_t pitch) {
    
    const unsigned int tidb = threadIdx.x;
    const unsigned int ti = blockIdx.x*blockDim.x + tidb;
    unsigned long int unit = 1;
    
    // Each thread compute a single line of the data
    if (ti < nbLine) {
		char* line = (char *)((char*)d_data + ti * pitch);
		int index = 0;
		int currentLetter = line[index];

        // Each char is converted to int and adds a unit to the corresponding index in the global memory
		while (currentLetter > 0) {
	    	atomicAdd(&d_result[currentLetter], unit);
	    	index++;
	    	currentLetter = line[index];
		}
    }
}
                            
////////////////////////////////////////////////////////////////////////////////
//! Kernel function to execute the computation in threads using Shared & Global Memory
//! @param d_data  input data in global memory
//! @param d_result  output result as array in global memory
//! @param nbLine  input size of the data in global memory
//! @param pitch  input pitch size of in the data global memory
////////////////////////////////////////////////////////////////////////////////

__global__ 
void kernelSharedMem(char* d_data, unsigned long int* d_result, int nbLine, size_t pitch) {
    
    const unsigned int tidb = threadIdx.x;
    const unsigned int ti = blockIdx.x*blockDim.x + tidb;
    unsigned long int zero = 0; 
    unsigned long int unit = 1;

    // Declare shared memory for result computation
    __shared__ unsigned long int s_result[NB_ASCII];
    // Reset shared memory values
    if (tidb == 0) {
        for (int i = 0; i < NB_ASCII; i++) {
            s_result[i] = zero;
        }
    }

    __syncthreads();
    
    // Each thread compute a single line of the data
    if (ti < nbLine) {
		char* line = (char *)((char*)d_data + ti * pitch);
		int index = 0;
		int currentLetter = line[index];

        // Each char is converted to int and adds a unit to the corresponding index in the shared memory
		while (currentLetter > 0) {
	    	atomicAdd(&s_result[currentLetter], unit);
	    	index++;
	    	currentLetter = line[index];
		}

        __syncthreads();

        // Each first thread of a bloc add the results of its bloc to the global memory 
        if (tidb == 0) {
            for (int i = 0; i < NB_ASCII; i++) {
                atomicAdd(&d_result[i], s_result[i]);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
	
    // Process the arguments of the call
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

    // Start timer
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Do the computation
	generateHisto(inputFileName, outputFileName);

    // Stop timer
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
    unsigned int resultSize = NB_ASCII * sizeof(unsigned long int);
    unsigned int lineSize = MAX_CHAR * sizeof(char);

    // Load input file
    FILE *inputFile = NULL;
    inputFile = fopen(inputFileName, "r");
    if (!inputFile) {
        printf("Wrong input file\n");
        printHelper();
		exit(EXIT_FAILURE);
    }

    // Allocate device memory
    char* d_data;
    unsigned long int* d_result;
    size_t pitch;
    checkCudaErrors(cudaMallocPitch((void **) &d_data, &pitch, lineSize, MAX_LINE));
    checkCudaErrors(cudaMalloc((void **) &d_result, resultSize));

    // Allocate host memory
    char h_data[MAX_LINE][MAX_CHAR];
    unsigned long int resultStorage[NB_ASCII];
    char str[MAX_CHAR];
    int nbLine = 0;
    int batchNum = 1;
    
    // Iterate over the file's lines
    while (fgets(str, MAX_CHAR, inputFile)) {
	
        // Batch size reached, send data to kernel for process
		if (nbLine == MAX_LINE) {

            printf("Batch N°%i: %i lines. \n", batchNum, nbLine);
	    	processBatchInKernel(&d_data, h_data, nbLine, pitch, lineSize, &d_result, resultSize, resultStorage);
            
            nbLine = 0;
            batchNum++;
		}

        // Add current line to the Batch
        strcpy(h_data[nbLine], str);
        nbLine++;
    }
    
    // Process last Batch (< MAX_LINE lines)
    printf("Batch N°%i: %i lines. \n", batchNum, nbLine);
    processBatchInKernel(&d_data, h_data, nbLine, pitch, lineSize, &d_result, resultSize, resultStorage);
    
    fclose(inputFile);
    
    // Write the output
    writeOutputCSV(resultStorage, outputFileName);

    // Cleanup memory
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_result));
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

void processBatchInKernel(  char** d_data,
                            char h_data[MAX_LINE][MAX_CHAR],
                            int nbLine,
                            size_t pitch,
                            int lineSize,
                            unsigned long int** d_result,
                            int resultSize,
                            unsigned long int resultStorage[NB_ASCII]) {
    // Allocate host memory for result
    unsigned long int h_result[NB_ASCII];

    // Setup execution parameters
    dim3  grid((nbLine + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
    dim3  threads(THREADS_PER_BLOCK, 1, 1);

    // Copy data to device
    checkCudaErrors(cudaMemcpy2D(*d_data, pitch, h_data, lineSize, lineSize, MAX_LINE, cudaMemcpyHostToDevice));
    
    // Execute the kernel
    // SWITCH THE COMMENT TO USE A DIFFERENT METHOD
    kernelSharedMem<<< grid, threads, 0 >>>(*d_data, *d_result, nbLine, pitch);
    // kernelGlobalMem<<< grid, threads, 0 >>>(*d_data, *d_result, nbLine, pitch);

    getLastCudaError("Kernel execution failed");
    
    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(&h_result, *d_result, resultSize, cudaMemcpyDeviceToHost));

    // Copy the result into resultStorage
    for (int index = 0; index < NB_ASCII; index++) {
        resultStorage[index] = h_result[index];
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Write the given output to the CSV file
//! @param result input the given ouput of the computations as an array of int
//! @param outputFileName input file name to write in
////////////////////////////////////////////////////////////////////////////////

void writeOutputCSV(unsigned long int result[NB_ASCII], char* outputFileName) {

    // Load output file
	FILE *outputFile;
	char asciiChar;
    if (outputFileName) {
        outputFile = fopen(outputFileName, "w+");
    } else {
        outputFile = fopen("out.csv", "w+");
    }
	
    // Write the result
	for (int index = FIST_RELEVANT_ASCII; index <= LAST_RELEVANT_ASCII; index++) {

        if (index >= FIRST_UPP_ASCII && index <= LAST_UPP_ASCII) {
            // Add uppercase count to char count
            result[index + FIST_RELEVANT_ASCII] += result[index];
        } else {
            // Write count in file
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