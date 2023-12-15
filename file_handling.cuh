#ifndef FILE_HAND_H
#define FILE_HAND_H
#define MAX_LINE_SIZE 1024
#include <stdio.h>
#include <stdlib.h>


//Struct for a singl input for black scholes
typedef struct {
  double S; // asset price(stock)
  double K; // strike price
  double T; // maturity
  double r; // risk free
  double skip; //skip
  double sigma; // volatility
  double skip2; // skip
} bs_inputs_t;

//struct for a buffer of inputs
typedef struct{
    long size;
    bs_inputs_t* list;
}input_list_t;


/**
* This helper function takes a file and returns all the data already feeded to the struct 
*
* @param *File
* @return input_list_t*
*
*/
input_list_t *read_input(FILE *file) {
  
  //initial size for allocation
  int vec_len = 100;
  input_list_t *blackScholes_inputs = (input_list_t*)malloc(sizeof(input_list_t));
  blackScholes_inputs->list = (bs_inputs_t*)malloc(vec_len*sizeof(bs_inputs_t));
  blackScholes_inputs->size=vec_len;

  // Skip the header line
  char header[MAX_LINE_SIZE];
  if (fgets(header, MAX_LINE_SIZE, file) == NULL) {
    perror("Error reading header");
    return NULL;
  }

  //row index
  long i = 0;
  //iterate until the end of the file
  while (feof(file)==0) {
    if (i >= blackScholes_inputs->size) {
        blackScholes_inputs->size*=2;
      blackScholes_inputs->list =
          (bs_inputs_t*)realloc(blackScholes_inputs->list, sizeof(bs_inputs_t) * blackScholes_inputs->size);
    }
    // for some reason every other row comes in a weird orther
    int count= fscanf(file, "%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", &blackScholes_inputs->list[i].S,
           &blackScholes_inputs->list[i].K, &blackScholes_inputs->list[i].T,
           &blackScholes_inputs->list[i].skip, &blackScholes_inputs->list[i].sigma,
           &blackScholes_inputs->list[i].r, &blackScholes_inputs->list[i].skip2);

    i++;
  }
   blackScholes_inputs->list = (bs_inputs_t*)realloc(blackScholes_inputs->list, sizeof(bs_inputs_t) *  i);
   blackScholes_inputs->size=i;
  return blackScholes_inputs;
}
#endif