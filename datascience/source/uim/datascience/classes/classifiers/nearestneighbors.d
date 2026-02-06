module uim.datascience.classes.classifiers.nearestneighbors;

import uim.datascience;
@safe:
/**
 * K-Nearest Neighbors Classifier
 */
class KNearestNeighbors {
  private {
    double[][] X_train;
    size_t[] y_train;
    size_t k;
  }

  /// Create KNN classifier
  this(size_t num_neighbors = 3) {
    assert(num_neighbors > 0, "k must be positive");
    k = num_neighbors;
  }

  /// Fit KNN (just stores training data)
  void fit(double[][] X, size_t[] y) {
    assert(X.length == y.length, "X and y must have same length");
    X_train = X.dup;
    y_train = y.dup;
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
    // Compute distances to all training samples
    struct Distance {
      double dist;
      size_t idx;
    }
    
    Distance[] distances = new Distance[X_train.length];
    foreach(i; 0 .. X_train.length) {
      double dist = 0.0;
      foreach(j; 0 .. sample.length) {
        double diff = sample[j] - X_train[i][j];
        dist += diff * diff;
      }
      distances[i] = Distance(sqrt(dist), i);
    }
    
    // Sort and get k nearest
    sort(distances);
    
    size_t[size_t] votes;
    foreach(i; 0 .. min(k, distances.length)) {
      size_t label = y_train[distances[i].idx];
      votes[label]++;
    }
    
    // Return most common label
    size_t maxVotes = 0;
    size_t prediction = 0;
    foreach(label, count; votes) {
      if (count > maxVotes) {
        maxVotes = count;
        prediction = label;
      }
    }
    
    return prediction;
  }
}