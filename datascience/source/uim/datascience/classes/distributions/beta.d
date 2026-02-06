/****************************************************************************************************************
* Copyright: © 2018-2026 Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*) 
* License: Subject to the terms of the Apache 2.0 license, as written in the included LICENSE.txt file. 
* Authors: Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*)
*****************************************************************************************************************/
module uim.datascience.classes.distributions.beta;

import uim.datascience;
@safe:

/**
 * Beta distribution
 */
class BetaDistribution {
  
  private double alpha;
  private double beta_param;

  this(double a, double b) pure {
    assert(a > 0.0 && b > 0.0, "Parameters must be positive");
    alpha = a;
    beta_param = b;
  }

  /// Probability density function
  double pdf(double x) const pure {
    if (x <= 0.0 || x >= 1.0) return 0.0;
    return pow(x, alpha - 1.0) * pow(1.0 - x, beta_param - 1.0) / beta(alpha, beta_param);
  }

  /// Cumulative distribution function (incomplete beta function)
  double cdf(double x) const pure {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    // Simplified approximation
    return incompleteBeta(x, alpha, beta_param);
  }

  /// Quantile function (approximation)
  double quantile(double p) const pure {
    assert(p >= 0.0 && p <= 1.0, "p must be between 0 and 1");
    if (p == 0.0) return 0.0;
    if (p == 1.0) return 1.0;
    return 0.5; // TODO: Implement proper quantile
  }
}
