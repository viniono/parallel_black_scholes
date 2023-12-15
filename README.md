# Parallalel Option Pricing in the GPU - Final Project CSC 213

## Students: 
    Diogo Tandeta Tartarotti
    Vinicius Ono Sant'anna

## Source code
    - main.cu: has the GPU kernel and main function.
    - black.cuh: has the functions used to calculate D1 and D2
    - file_handling.cuh: has the function to process input from .csv files
    - time_util.cuh: has the function to time the kernel

## Usage
    To build the program, run `make`.
    To run the program, run `./main path/to/datafile`
    To see results from the program, open `prices_output.csv`
s
    