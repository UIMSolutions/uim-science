/****************************************************************************************************************
* Copyright: © 2018-2026 Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*) 
* License: Subject to the terms of the Apache 2.0 license, as written in the included LICENSE.txt file. 
* Authors: Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*)
*****************************************************************************************************************/
module uim.datascience.classes.distributions.chisquared;

import uim.datascience;
@safe:

/**
 * Chi-squared distribution
 */
class ChiSquaredDistribution {
  
  private uint df;

  this(uint degrees_of_freedom) pure {
    assert(degrees_of_freedom > 0, "Degrees of freedom must be positive");
    df = degrees_of_freedom;
  }

  /// Probability density function
  double pdf(double x) const pure {
    if (x <= 0.0) return 0.0;
    double k = df / 2.0;
    return pow(x, k - 1.0) * exp(-x / 2.0) / (pow(2.0, k) * exp(lgamma(k)));
  }

  /// Cumulative distribution function (approximation)
  double cdf(double x) const pure {
    if (x <= 0.0) return 0.0;
    return incompleteBeta(x / (x + 2.0), df / 2.0, 1.0);
  }
}
