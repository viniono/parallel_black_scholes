// #include "black.c"
#include "black.h"
#include <stdio.h>
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

int main() {

  bs_inputs blackScholes_inputs[10000];

  FILE *file = fopen("data/UKX_Calls.csv", "r");

  if (file == NULL) {
    perror("Error opening file");
    return 1;
  }

  // Skip the header line
  char header[MAX_LINE_SIZE];
  if (fgets(header, MAX_LINE_SIZE, file) == NULL) {
    perror("Error reading header");
    fclose(file);
    return 1;
  }

  int i = 0;
  while (!feof(file)) {
    if (i == 30)
      break;
    // for some reason every other row comes in a weird orther
    fscanf(file, "%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", &blackScholes_inputs[i].S,
           &blackScholes_inputs[i].K, &blackScholes_inputs[i].T,
           &blackScholes_inputs[i].skip, &blackScholes_inputs[i].sigma,
           &blackScholes_inputs[i].r, &blackScholes_inputs[i].skip2);
    i++;
  }
  fclose(file);

  for (; i > 0; i--) {
    double call = BS_CALL(blackScholes_inputs[i].S, blackScholes_inputs[i].K,
                          blackScholes_inputs[i].T, blackScholes_inputs[i].r,
                          blackScholes_inputs[i].sigma);
    double put = BS_PUT(blackScholes_inputs[i].S, blackScholes_inputs[i].K,
                        blackScholes_inputs[i].T, blackScholes_inputs[i].r,
                        blackScholes_inputs[i].sigma);
    printf("CALL fair: %lf, PUT fair:  %lf\n", call, put);
  }
}
