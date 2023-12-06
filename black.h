#ifndef BLACK_H
#define BLACK_H
/* double erf(double x); */
// Normal CDF using error function
double cdf(double x);

// D1 Black Scholes
double D1(double S, double K, double T, double r, double sigma);
// D2 Black Scholes
double D2(double d1, double sigma, double T);

// BLACK CALL
double BS_CALL(double S, double K, double T, double r, double sigma);

// BLACK PUT
double BS_PUT(double S, double K, double T, double r, double sigma);
#endif
