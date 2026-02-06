/****************************************************************************************************************
* Copyright: © 2018-2026 Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*) 
* License: Subject to the terms of the Apache 2.0 license, as written in the included LICENSE.txt file. 
* Authors: Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*)
*****************************************************************************************************************/
module uim.datascience.classes.distributions.normal;

import uim.datascience;
@safe:

/**
 * Normal (Gaussian) distribution
 */
class NormalDistribution {
  
  private double mean;
  private double stddev;

  this(double m = 0.0, double sd = 1.0) pure {
    assert(sd > 0.0, "Standard deviation must be positive");
    mean = m;
    stddev = sd;
  }

  /// Probability density function
  double pdf(double x) const pure {
    double z = (x - mean) / stddev;
    return (1.0 / (stddev * sqrt(2 * PI))) * exp(-(z * z) / 2.0);
  }

  /// Cumulative distribution function (approximation)
  double cdf(double x) const pure {
    double z = (x - mean) / stddev;
    return 0.5 * (1.0 + erf(z / sqrt(2.0)));
  }

  /// Quantile function
  double quantile(double p) const pure {
    assert(p > 0.0 && p < 1.0, "p must be between 0 and 1");
    // Approximation using the rational approximation method
    if (p < 0.5) {
      double t = sqrt(-2.0 * log(p));
      return mean + stddev * (t - (2.515517 + 0.802853*t + 0.010328*t*t) / 
             (1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t));
    } else {
      return mean - stddev * quantile(1.0 - p);
    }
  }

  /// Generate random samples
  double[] sample(size_t n) const {
    double[] samples = new double[n];
    foreach(i; 0 .. n) {
      // Box-Muller transform
      double u1 = uniform(0.0, 1.0);
      double u2 = uniform(0.0, 1.0);
      double z0 = sqrt(-2.0 * log(u1)) * cos(2 * PI * u2);
      samples[i] = mean + stddev * z0;
    }
    return samples;
  }
}
