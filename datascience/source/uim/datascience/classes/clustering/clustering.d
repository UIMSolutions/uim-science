/****************************************************************************************************************
* Copyright: © 2018-2026 Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*) 
* License: Subject to the terms of the Apache 2.0 license, as written in the included LICENSE.txt file. 
* Authors: Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*)
*****************************************************************************************************************/
module uim.datascience.clustering;

import std.math;
import std.algorithm;
import std.array;
import uim.core;

/**
 * K-Means clustering algorithm
 */
class KMeans {
  private {
    size_t k;
    double[][] centers;
    size_t[] labels;
    double inertia;
  }

  /// Create K-Means model
  this(size_t n_clusters) {
    assert(n_clusters > 0, "Number of clusters must be positive");
    k = n_clusters;
  }

  /// Fit the model
  void fit(double[][] data, size_t max_iterations = 100) {
    assert(data.length > 0 && data[0].length > 0, "Data must not be empty");
    assert(k <= data.length, "Number of clusters cannot exceed number of samples");
    
    size_t n_samples = data.length;
    size_t n_features = data[0].length;
    
    // Initialize centers randomly
    centers = new double[][](k, n_features);
    foreach(i; 0 .. k) {
      foreach(j; 0 .. n_features) {
        centers[i][j] = data[i][j];
      }
    }
    
    labels = new size_t[n_samples];
    
    foreach(iter; 0 .. max_iterations) {
      // Assign points to nearest center
      foreach(i; 0 .. n_samples) {
        double minDist = double.max;
        size_t bestCluster = 0;
        
        foreach(c; 0 .. k) {
          double dist = 0.0;
          foreach(j; 0 .. n_features) {
            double diff = data[i][j] - centers[c][j];
            dist += diff * diff;
          }
          
          if (dist < minDist) {
            minDist = dist;
            bestCluster = c;
          }
        }
        labels[i] = bestCluster;
      }
      
      // Update centers
      double[][] newCenters = new double[][](k, n_features);
      size_t[] counts = new size_t[k];
      
      foreach(i; 0 .. n_samples) {
        size_t c = labels[i];
        counts[c]++;
        foreach(j; 0 .. n_features) {
          newCenters[c][j] += data[i][j];
        }
      }
      
      foreach(c; 0 .. k) {
        if (counts[c] > 0) {
          foreach(j; 0 .. n_features) {
            newCenters[c][j] /= counts[c];
          }
        } else {
          foreach(j; 0 .. n_features) {
            newCenters[c][j] = centers[c][j];
          }
        }
      }
      
      centers = newCenters;
    }
    
    // Calculate inertia (within-cluster sum of squares)
    inertia = 0.0;
    foreach(i; 0 .. n_samples) {
      double dist = 0.0;
      foreach(j; 0 .. n_features) {
        double diff = data[i][j] - centers[labels[i]][j];
        dist += diff * diff;
      }
      inertia += dist;
    }
  }

  /// Predict cluster labels
  size_t[] predict(double[][] data) const {
    assert(data.length > 0, "Data must not be empty");
    
    size_t n_samples = data.length;
    size_t n_features = data[0].length;
    size_t[] predictions = new size_t[n_samples];
    
    foreach(i; 0 .. n_samples) {
      double minDist = double.max;
      size_t bestCluster = 0;
      
      foreach(c; 0 .. k) {
        double dist = 0.0;
        foreach(j; 0 .. n_features) {
          double diff = data[i][j] - centers[c][j];
          dist += diff * diff;
        }
        
        if (dist < minDist) {
          minDist = dist;
          bestCluster = c;
        }
      }
      predictions[i] = bestCluster;
    }
    
    return predictions;
  }

  /// Get cluster centers
  @property double[][] get_centers() const { return centers.dup; }

  /// Get inertia
  @property double get_inertia() const { return inertia; }
}

