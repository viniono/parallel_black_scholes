#ifndef BLACK_H
#define BLACK_H

// //Link for relation of ERF and cdf
// //https://www.johndcook.com/erf_and_normal_cdf.pdf

// // Error Function
__device__ double erf(double x) {
  // 	// constants for error function
  // //source for optimized constants:
  // //
  // https://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf/
  double a1 = 0.254829592;
  double a2 = -0.284496736;
  double a3 = 1.421413741;
  double a4 = -1.453152027;
  double a5 = 1.061405429;
  double p = 0.3275911;
  int sign = 1;
  if (x < 0) {
    sign = -1;
  }
  x = fabs(x);

  double t = 1.0 / (1.0 + p * x);

  double y =
      1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

  return sign * y;
}

// Normal CDF using error function
__device__ double cdf(double x) {
  double SQRT_2 = 1.41421356237;
  return (erf(x / SQRT_2) + 1.0) / 2.0;
}

// D1 Black Scholes
__device__ double D1(double S, double K, double T, double r, double sigma) {
  return (log(S / K) + (r + pow(sigma, 2) / 2)) / (sigma * sqrt(T));
}

// D2 Black Scholes
__device__ double D2(double d1, double sigma, double T) {
  return d1 - (sigma * sqrt(T));
}

// BLACK CALL
__device__ double BS_CALL(double S, double K, double T, double r,
                          double sigma) {
  double d1 = D1(S, K, T, r, sigma);
  double d2 = D2(d1, sigma, T);
  return S * cdf(d1) - K * __expf(-r * T) * cdf(d2);
}

// BLACK PUT
__device__ double BS_PUT(double S, double K, double T, double r, double sigma) {
  double d1 = D1(S, K, T, r, sigma);
  double d2 = D2(d1, sigma, T);
  return K * __expf(-r * T) * cdf(-d2) - S * cdf(-d1);
}

#endif
