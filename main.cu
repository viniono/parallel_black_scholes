#include "black.cuh"
#include "file_handling.cuh"
#include "time_util.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#define THREADS_PER_BLOCK 64
#define MAX_LINE_SIZE 1024

typedef struct {
  double call_price;
  double put_price;
} option_price_t;

__global__ void pricer(bs_inputs_t *blackScholes_inputs,
                       option_price_t *prices, long N) {
  // Which index of the array should this thread use?
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  //checking if out of bounds
  if(index>=N){
    return;
  }
  // Unmarshall struct
  double K = blackScholes_inputs[index].K;
  double S = blackScholes_inputs[index].S;
  double r = blackScholes_inputs[index].r;
  double T = blackScholes_inputs[index].T;
  double sigma = blackScholes_inputs[index].sigma;
  // Compute d1 and d2 in each thread once only (making sure we have no repeats) since both call and put calculate it
  double d1 = D1(S, K, T, r, sigma);
  double d2 = D2(d1, sigma, T);
  prices[index].call_price = S * cdf(d1) - K * __expf(-r * T) * cdf(d2);
  prices[index].put_price = K * __expf(-r * T) - S + prices[index].call_price;
}

int main(int argc, char **argv) {
  
  FILE *file = fopen(argv[1], "r");
  
  long N;
  input_list_t *input_list;
  if (file == NULL) {
    perror("Error opening file");
    exit(2);
  }

  input_list = read_input(file);
  fclose(file);

  if (input_list == NULL) {
    perror("Error reading input");
    exit(2);
  }

  N = input_list->size;
  // Allocate arrays for X and Y on the CPU. This memory is only usable on the
  // CPU
  bs_inputs_t *CPU_blackScholes_inputs = input_list->list;
  option_price_t CPU_prices[N];

  // GPU
  bs_inputs_t *GPU_blackScholes_inputs;
  option_price_t *GPU_prices;

  if (cudaMalloc(&GPU_blackScholes_inputs, sizeof(bs_inputs_t) * N) !=
      cudaSuccess) {
    fprintf(stderr, "Failed to allocate X array on GPU\n");
    exit(2);
  }

  // Allocate space for the x array on the GPU
  if (cudaMalloc(&GPU_prices, sizeof(option_price_t) * N) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate X array on GPU\n");
    exit(2);
  }

  // Copy the cpu's x array to the gpu with cudaMemcpy
  if (cudaMemcpy(GPU_blackScholes_inputs, CPU_blackScholes_inputs,
                 sizeof(bs_inputs_t) * N,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy X to the GPU\n");
  }

  // Calculate the number of blocks to run, rounding up to include all threads
  size_t blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  // Start timing the performance
  size_t start_time = time_micros();
  
  
  // Run the pricer kernel
  pricer<<<blocks, THREADS_PER_BLOCK>>>(GPU_blackScholes_inputs, GPU_prices, N);

  // Wait for the kernel to finish
  if (cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n",
            cudaGetErrorString(cudaPeekAtLastError()));
  }
  // Calculate the elapsed time in miliseconds
  size_t elapsed_time = time_micros() - start_time;
  double seconds = (double)elapsed_time / 1000000;
  double computing_rate = (double)N / seconds;
  printf("Number of options: %d\n", N);
  printf("Total computation time: %lu\u03BCs\n", elapsed_time);
  printf("Computation rate: %.2lf options per second\n", computing_rate);
  // Copy the y array back from the gpu to the cpu
  if (cudaMemcpy(CPU_prices, GPU_prices, sizeof(option_price_t) * N,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy Y from the GPU\n");
  }

  FILE *output_file = fopen("prices_output.csv", "w");

  if (output_file == NULL) {
    perror("Error opening file");
    return 1;
  }
  //Column names
  fprintf(output_file, "call_prices,put_prices\n");
  // Print the updated y array
  for (int i = 0; i < N; i++) {
    // printf("%d: %f\n",i,CPU_prices[i]);
    fprintf(output_file, "%f,%f\n", CPU_prices[i].call_price,
            CPU_prices[i].put_price);
  }

  fclose(output_file);
  free(input_list);
  cudaFree(GPU_prices);
  cudaFree(GPU_blackScholes_inputs);
  free(CPU_blackScholes_inputs);
  return 0;
}
