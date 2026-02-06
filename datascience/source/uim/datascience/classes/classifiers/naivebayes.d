module uim.datascience.classes.classifiers.naivebayes;

import uim.datascience;
@safe:

/**
 * Naive Bayes Classifier
 */
class NaiveBayesClassifier {
  private {
    struct ClassStats {
      double prior;
      double[] mean;
      double[] var;
    }
    
    ClassStats[size_t] class_stats;
  }

  /// Fit Naive Bayes
  void fit(double[][] X, size_t[] y) {
    assert(X.length == y.length, "X and y must have same length");
    assert(X.length > 0 && X[0].length > 0, "Data must not be empty");
    
    size_t n_samples = X.length;
    size_t n_features = X[0].length;
    
    // Group samples by class
    double[][size_t] class_samples;
    size_t[size_t] class_counts;
    
    foreach(i; 0 .. n_samples) {
      size_t label = y[i];
      class_samples[label] ~= i;
      class_counts[label]++;
    }
    
    // Calculate statistics for each class
    foreach(label; class_counts.keys.sort()) {
      auto indices = class_samples[label];
      ClassStats stats;
      stats.prior = cast(double)indices.length / n_samples;
      stats.mean = new double[n_features];
      stats.var = new double[n_features];
      
      // Calculate mean and variance for each feature
      foreach(j; 0 .. n_features) {
        double sum = 0.0;
        foreach(idx; indices) {
          sum += X[idx][j];
        }
        stats.mean[j] = sum / indices.length;
        
        double var = 0.0;
        foreach(idx; indices) {
          double diff = X[idx][j] - stats.mean[j];
          var += diff * diff;
        }
        stats.var[j] = var / (indices.length - 1);
      }
      
      class_stats[label] = stats;
    }
  }

  /// Predict class labels
  size_t[] predict(double[][] X) const {
    assert(X.length > 0, "Data must not be empty");
    
    size_t[] predictions = new size_t[X.length];
    foreach(i; 0 .. X.length) {
      predictions[i] = predictSample(X[i]);
    }
    return predictions;
  }

  private size_t predictSample(double[] sample) const {
    double maxProb = -double.max;
    size_t prediction = 0;
    
    foreach(label, stats; class_stats) {
      double prob = log(stats.prior);
      
      foreach(j; 0 .. sample.length) {
        double mean = stats.mean[j];
        double var = stats.var[j];
        double num = exp(-((sample[j] - mean) ^^ 2) / (2 * var));
        double denom = sqrt(2 * PI * var);
        prob += log(num / denom);
      }
      
      if (prob > maxProb) {
        maxProb = prob;
        prediction = label;
      }
    }
    
    return prediction;
  }
}
