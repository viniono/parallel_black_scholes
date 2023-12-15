# Parallalel Option Pricing in the GPU - Final Project CSC 213

## Students: 
Diogo Tandeta Tartarotti and Vinicius Ono Sant'anna

## Source code
- main.cu: has the GPU kernel and main function.
- black.cuh: has the functions used to calculate D1 and D2
- file_handling.cuh: has the function to process input from .csv files
- time_util.cuh: has the function to time the kernel

## Data Format
.csv file with the following comma-separated columns in this order: Stock Price, Strike Price, Maturity, Dividends, Volatility, Risk-free, Call Price.
Also, the first line of the file must not contain more than 1024 characters.

## Usage
This program uses the European options solution of the Black-Scholes model to find the call and put prices of provided options. The program then writes the prices to an output file and reports statistics about its execution to the terminal. There are input data files of varying sizes in the `data` folder. 

To build the program, run `make`.
To run the program, run `./main path/to/datafile`
To see the results of the program, open `prices_output.csv` in a text editor

Example: 
```
> ./main data/SNP.csv
> Number of options: 57516
Total computation time: 688μs
Computation rate: 83598837.21 options per second
> vim prices_output.csv
```
