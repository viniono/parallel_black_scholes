#ifndef BLACK_H
#define BLACK_H

//Link for relation of ERF and cdf 
//https://www.johndcook.com/erf_and_normal_cdf.pdf

// Normal CDF using error function
__device__ double cdf(double x) {
	double SQRT_2= 1.41421356237;
    return (erf(x / SQRT_2) + 1.0) / 2.0;
}

// D1 Black Scholes
__device__ double D1(double S, double K, double T, double r, double sigma) {
	return (log(S/K) + (r + pow(sigma, 2)/2)) / (sigma * sqrt(T));
}

// D2 Black Scholes
__device__ double D2(double d1, double sigma, double T) {
	return d1 - (sigma * sqrt(T));
}

// BLACK CALL
__device__ double BS_CALL(double S, double K, double T, double r, double sigma) {
	double d1 = D1(S, K, T, r, sigma);
	double d2 = D2(d1, sigma, T);
	return S * cdf(d1) - K * __expf(-r*T) * cdf(d2);
}

// BLACK PUT
__device__ double BS_PUT(double S, double K, double T, double r, double sigma) {
	double d1 = D1(S, K, T, r, sigma);
	double d2 = D2(d1, sigma, T);
	return  K * __expf(-r*T) * cdf(-d2) - S * cdf(-d1);
}
#endif
