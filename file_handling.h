#ifndef FILE_HAND_H
#define FILE_HAND_H
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

bs_inputs *read_input(FILE *file);
#endif
