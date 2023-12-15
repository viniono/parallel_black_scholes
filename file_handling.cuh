#ifndef FILE_HAND_H
#define FILE_HAND_H
#define MAX_LINE_SIZE 1024
#include <stdio.h>
#include <stdlib.h>

// Struct to store input data
typedef struct {
  double S;     // asset price(stock)
  double K;     // strike price
  double T;     // maturity
  double r;     // risk free
  double skip;  // skip
  double sigma; // volatility
  double skip2; // skip
} bs_inputs_t;

// struct for a buffer of inputs
typedef struct {
  long size;
  bs_inputs_t *list;
} input_list_t;

/**
 * This helper function takes a file and returns all the data already feeded to
 * the struct
 *
 * @param *File
 * @return input_list_t*
 *
 */
input_list_t *read_input(FILE *file) {

  // initial size for allocation
  int vec_len = 100;
  input_list_t *black_scholes_inputs =
      (input_list_t *)malloc(sizeof(input_list_t));
  black_scholes_inputs->list =
      (bs_inputs_t *)malloc(vec_len * sizeof(bs_inputs_t));
  black_scholes_inputs->size = vec_len;

  // Skip the header line
  char header[MAX_LINE_SIZE];
  if (fgets(header, MAX_LINE_SIZE, file) == NULL) {
    perror("Error reading header");
    return NULL;
  }

  // row index
  long i = 0;
  // iterate until the end of the file
  while (feof(file) == 0) {
    if (i >= black_scholes_inputs->size) {
      black_scholes_inputs->size *= 2;
      black_scholes_inputs->list = (bs_inputs_t *)realloc(
          black_scholes_inputs->list,
          sizeof(bs_inputs_t) * black_scholes_inputs->size);
    }
    // for some reason every other row comes in a weird orther
    int count = fscanf(
        file, "%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", &black_scholes_inputs->list[i].S,
        &black_scholes_inputs->list[i].K, &black_scholes_inputs->list[i].T,
        &black_scholes_inputs->list[i].skip, &black_scholes_inputs->list[i].sigma,
        &black_scholes_inputs->list[i].r, &black_scholes_inputs->list[i].skip2);

    i++;
  }
  black_scholes_inputs->list = (bs_inputs_t *)realloc(black_scholes_inputs->list,
                                                     sizeof(bs_inputs_t) * i);
  black_scholes_inputs->size = i;
  return black_scholes_inputs;
}
#endif
