// #include "black.c"
#include "black.h"
#include <stdio.h>
#include <stdlib.h>
typedef struct {
  double S; // asset price(stock)
  double K; // strike price
  double T; // maturity
  double r; // risk free
  double skip;
  double sigma; // volatility
  double skip2; // volatility
} bs_inputs;

#define MAX_LINE_SIZE 1024

bs_inputs *read_input(FILE *file) {
  int vec_len = 100;
  bs_inputs *blackScholes_inputs = malloc(vec_len * sizeof(bs_inputs));

  // Skip the header line
  char header[MAX_LINE_SIZE];
  if (fgets(header, MAX_LINE_SIZE, file) == NULL) {
    perror("Error reading header");
    return NULL;
  }
  int i = 0;
  while (!feof(file)) {
    if (i >= vec_len) {
      vec_len *= 2;
      blackScholes_inputs =
          realloc(blackScholes_inputs, sizeof(bs_inputs) * vec_len);
    }
    // for some reason every other row comes in a weird orther
    fscanf(file, "%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", &blackScholes_inputs[i].S,
           &blackScholes_inputs[i].K, &blackScholes_inputs[i].T,
           &blackScholes_inputs[i].skip, &blackScholes_inputs[i].sigma,
           &blackScholes_inputs[i].r, &blackScholes_inputs[i].skip2);
    i++;
  }
  blackScholes_inputs = realloc(blackScholes_inputs, sizeof(bs_inputs) * i);

  return blackScholes_inputs;
}
