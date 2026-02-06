/****************************************************************************************************************
* Copyright: © 2018-2026 Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*) 
* License: Subject to the terms of the Apache 2.0 license, as written in the included LICENSE.txt file. 
* Authors: Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*)
*****************************************************************************************************************/
module uim.datascience.helpers.distributions;

import uim.datascience;
@safe:

/**
 * Probability distribution functions
 */

/// Gamma function using Stirling's approximation
private double gamma(double x) pure {
  if (x < 0.5) {
    return PI / (sin(PI * x) * gamma(1.0 - x));
  }
  
  x -= 1.0;
  double a = 1.0;
  double t = x + 7.0;
  t = (2.506628277459 * sqrt(t) * pow(t, x + 0.5) * exp(-t)) / a;
  return t;
}

/// Beta function
private double beta(double a, double b) pure {
  return exp(lgamma(a) + lgamma(b) - lgamma(a + b));
}

/// Log-gamma function (better numerical stability)
private double lgamma(double x) pure {
  if (x <= 0.0) return double.nan;
  
  const double g = 7.0;
  const double[] coef = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ];

  x -= 1.0;
  double z = coef[0];
  foreach(i; 1 .. coef.length) {
    z += coef[i] / (x + i);
  }

  double t = x + g + 0.5;
  return 0.5 * log(2 * PI) + (x + 0.5) * log(t) - t + log(z);
}

/// Incomplete beta function (regularized)
private double incompleteBeta(double x, double a, double b) pure {
  if (x == 0.0) return 0.0;
  if (x == 1.0) return 1.0;

  // Continued fraction approximation
  double front = exp(a * log(x) + b * log(1.0 - x) - lgamma(a + b) + lgamma(a) + lgamma(b));
  
  if (x < (a + 1.0) / (a + b + 2.0)) {
    return front * continuedFraction(x, a, b) / a;
  } else {
    return 1.0 - front * continuedFraction(1.0 - x, b, a) / b;
  }
}

/// Helper for continued fraction approximation
private double continuedFraction(double x, double a, double b) pure {
  const int max_iter = 500;
  const double epsilon = 1e-10;

  double am = 1.0;
  double bm = 1.0;
  double az = 1.0;
  double qab = a + b;
  double qap = a + 1.0;
  double qam = a - 1.0;
  double bz = 1.0 - qab * x / qap;

  for (int m = 1; m <= max_iter; m++) {
    double em = m;
    double tem = 2.0 * em;
    double d = em * (b - em) * x / ((qam + tem) * (a + tem));
    double ap = az + d * am;
    double bp = bz + d * bm;
    d = -(a + em) * (qab + em) * x / ((qap + tem) * (a + tem));
    az = ap / bp;
    bz = 1.0;
    am = ap / bp;
    bm = d;
    if (abs(az - bz) < epsilon) {
      return az;
    }
  }

  return az;
}

/// Helper for uniform random number generation
private double uniform(double a, double b) @trusted {
  import std.random;
  return a + (b - a) * uniform01!(double)();
}
