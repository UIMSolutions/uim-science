/****************************************************************************************************************
* Copyright: © 2018-2026 Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*) 
* License: Subject to the terms of the Apache 2.0 license, as written in the included LICENSE.txt file. 
* Authors: Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*)
*****************************************************************************************************************/
module uim.datascience.classes.distributions.exponential;

import uim.datascience;
@safe:

/**
 * Exponential distribution
 */
class ExponentialDistribution {
  
  private double lambda;

  this(double rate = 1.0) pure {
    assert(rate > 0.0, "Rate must be positive");
    lambda = rate;
  }

  /// Probability density function
  double pdf(double x) const pure {
    if (x < 0.0) return 0.0;
    return lambda * exp(-lambda * x);
  }

  /// Cumulative distribution function
  double cdf(double x) const pure {
    if (x < 0.0) return 0.0;
    return 1.0 - exp(-lambda * x);
  }

  /// Quantile function
  double quantile(double p) const pure {
    assert(p >= 0.0 && p <= 1.0, "p must be between 0 and 1");
    return -log(1.0 - p) / lambda;
  }
}