/**
 * Hierarchical clustering with agglomerative approach
 */
class HierarchicalClustering {
  private {
    struct Cluster {
      size_t[] indices;
      double[] centroid;
    }
    
    Cluster[] clusters;
  }

  /// Perform hierarchical clustering
  void fit(double[][] data, size_t n_clusters) {
    assert(data.length > 0 && data[0].length > 0, "Data must not be empty");
    assert(n_clusters > 0 && n_clusters <= data.length, "Invalid number of clusters");
    
    size_t n_samples = data.length;
    size_t n_features = data[0].length;
    
    // Initialize each point as a cluster
    clusters = new Cluster[n_samples];
    foreach(i; 0 .. n_samples) {
// 
      clusters[i].indices = [i];
      clusters[i].centroid = new double[n_features];
      foreach(j; 0 .. n_features) {
        clusters[i].centroid[j] = data[i][j];
      }
    }
    
    // Merge clusters until n_clusters remain
    while (clusters.length > n_clusters) {
      double minDist = double.max;
      size_t mergeI = 0, mergeJ = 1;
      
      // Find two closest clusters
      foreach(i; 0 .. clusters.length) {
        foreach(j; i+1 .. clusters.length) {
          double dist = clusterDistance(clusters[i], clusters[j]);
          if (dist < minDist) {
            minDist = dist;
            mergeI = i;
            mergeJ = j;
          }
        }
      }
      
      // Merge clusters
      foreach(idx; clusters[mergeJ].indices) {
        clusters[mergeI].indices ~= idx;
      }
      
      // Update centroid
      size_t n_features = clusters[mergeI].centroid.length;
      double[] newCentroid = new double[n_features];
      foreach(idx; clusters[mergeI].indices) {
        foreach(j; 0 .. n_features) {
          newCentroid[j] += data[idx][j];
        }
      }
      foreach(j; 0 .. n_features) {
        newCentroid[j] /= clusters[mergeI].indices.length;
      }
      clusters[mergeI].centroid = newCentroid;
      
      // Remove merged cluster
      clusters = clusters[0 .. mergeJ] ~ clusters[mergeJ+1 .. $];
    }
  }

  /// Get cluster assignments
  size_t[] getLabels(size_t n_samples) const {
    size_t[] labels = new size_t[n_samples];
    foreach(c, cluster; clusters) {
      foreach(idx; cluster.indices) {
        labels[idx] = c;
      }
    }
    return labels;
  }

  private static double clusterDistance(Cluster c1, Cluster c2) pure {
    double dist = 0.0;
    foreach(i; 0 .. c1.centroid.length) {
      double diff = c1.centroid[i] - c2.centroid[i];
      dist += diff * diff;
    }
    return sqrt(dist);
  }
}
