/****************************************************************************************************************
* Copyright: © 2018-2026 Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*) 
* License: Subject to the terms of the Apache 2.0 license, as written in the included LICENSE.txt file. 
* Authors: Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*)
*****************************************************************************************************************/
module uim.datascience.classes.distributions.uniform;

import uim.datascience;
@safe:

/**
 * Uniform distribution
 */
class UniformDistribution {
  
  private double a;
  private double b;

  this(double min_val = 0.0, double max_val = 1.0) pure {
    assert(min_val < max_val, "min must be less than max");
    a = min_val;
    b = max_val;
  }

  /// Probability density function
  double pdf(double x) const pure {
    if (x < a || x > b) return 0.0;
    return 1.0 / (b - a);
  }

  /// Cumulative distribution function
  double cdf(double x) const pure {
    if (x < a) return 0.0;
    if (x > b) return 1.0;
    return (x - a) / (b - a);
  }

  /// Quantile function
  double quantile(double p) const pure {
    assert(p >= 0.0 && p <= 1.0, "p must be between 0 and 1");
    return a + p * (b - a);
  }
}