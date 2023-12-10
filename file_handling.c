#include <stdio.h>
#include <stdlib.h>
#include "file_handling.h"
#define MAX_LINE_SIZE 1024

input_list_t *read_input(FILE *file) {
    
  int vec_len = 100;
  input_list_t *blackScholes_inputs = malloc(sizeof(input_list_t));
  blackScholes_inputs->list = malloc(vec_len*sizeof(bs_inputs_t));
  blackScholes_inputs->size=vec_len;

  // Skip the header line
  char header[MAX_LINE_SIZE];
  if (fgets(header, MAX_LINE_SIZE, file) == NULL) {
    perror("Error reading header");
    return NULL;
  }
  int i = 0;
  while (!feof(file)) {
    if (i >= blackScholes_inputs->size) {
        blackScholes_inputs->size*=2;
      blackScholes_inputs->list =
          realloc(blackScholes_inputs->list, sizeof(bs_inputs_t) * blackScholes_inputs->size);
    }
    // for some reason every other row comes in a weird orther
    int count= fscanf(file, "%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", &blackScholes_inputs->list[i].S,
           &blackScholes_inputs->list[i].K, &blackScholes_inputs->list[i].T,
           &blackScholes_inputs->list[i].skip, &blackScholes_inputs->list[i].sigma,
           &blackScholes_inputs->list[i].r, &blackScholes_inputs->list[i].skip2);
        if(count==EOF||count==0){
            perror("fscanf");
        }
    i++;
  }
   blackScholes_inputs->list = realloc(blackScholes_inputs->list, sizeof(bs_inputs_t) *  i);
   blackScholes_inputs->size=i;
  return blackScholes_inputs;
}