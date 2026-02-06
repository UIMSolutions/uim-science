/****************************************************************************************************************
* Copyright: © 2018-2026 Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*) 
* License: Subject to the terms of the Apache 2.0 license, as written in the included LICENSE.txt file. 
* Authors: Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*)
*****************************************************************************************************************/
module uim.datascience.classes.classifiers.decisiontree;

import uim.datascience;
@safe:

/**
 * Decision Tree Classifier
 */
class DecisionTreeClassifier {
  private {
    struct Node {
      bool isLeaf;
      size_t featureIndex;
      double threshold;
      size_t class_label;
      Node* left;
      Node* right;
    }
    
    Node* root;
    size_t max_depth;
    size_t min_samples_split;
  }

  /// Create decision tree classifier
  this(size_t tree_max_depth = 10, size_t tree_min_samples = 2) {
    max_depth = tree_max_depth;
    min_samples_split = tree_min_samples;
    root = null;
  }

  /// Fit decision tree
  void fit(double[][] X, size_t[] y) {
    assert(X.length == y.length, "X and y must have same length");
    assert(X.length > 0 && X[0].length > 0, "Data must not be empty");
    
    size_t[] indices = new size_t[X.length];
    foreach(i; 0 .. X.length) indices[i] = i;
    
    root = buildTree(X, y, indices, 0);
  }

  /// Predict class labels
  size_t[] predict(double[][] X) const {
    assert(X.length > 0, "Data must not be empty");
    
    size_t[] predictions = new size_t[X.length];
    foreach(i; 0 .. X.length) {
      predictions[i] = predictSample(root, X[i]);
    }
    return predictions;
  }

  private Node* buildTree(double[][] X, size_t[] y, size_t[] indices, size_t depth) {
    auto node = new Node();
    
    // Base cases
    if (indices.length < min_samples_split || depth >= max_depth) {
      node.isLeaf = true;
      node.class_label = majorityClass(y, indices);
      return node;
    }
    
    // All same class
    size_t label = y[indices[0]];
    bool allSame = true;
    foreach(idx; indices) {
      if (y[idx] != label) {
        allSame = false;
        break;
      }
    }
    if (allSame) {
      node.isLeaf = true;
      node.class_label = label;
      return node;
    }
    
    // Find best split
    double bestGain = -1.0;
    size_t bestFeature = 0;
    double bestThreshold = 0.0;
    
    // Loop over features
    foreach(j; 0 .. X[0].length) {
      double[] values = new double[indices.length];
      foreach(i, idx; indices) {
        values[i] = X[idx][j];
      }
      
      // Sort values for potential splits
      sort(values);
      
      // Try splits between unique values
      foreach(i; 0 .. values.length - 1) {
        double threshold = (values[i] + values[i+1]) / 2.0;
        
        // Split data
        size_t[] leftIndices, rightIndices;
        foreach(idx; indices) {
          if (X[idx][j] <= threshold) {
            leftIndices ~= idx;
          } else {
            rightIndices ~= idx;
          }
        }
        
        // Skip invalid splits
        if (leftIndices.length == 0 || rightIndices.length == 0) continue;
        
        // Calculate information gain
        double gain = informationGain(y, indices, leftIndices, rightIndices);
        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = j;
          bestThreshold = threshold;
        }
      }
    }
    
    if (bestGain < 0.0) {
      node.isLeaf = true;
      node.class_label = majorityClass(y, indices);
      return node;
    }
    
    // Split data
    size_t[] leftIndices, rightIndices;
    foreach(idx; indices) {
      if (X[idx][bestFeature] <= bestThreshold) {
        leftIndices ~= idx;
      } else {
        rightIndices ~= idx;
      }
    }
    
    node.isLeaf = false;
    node.featureIndex = bestFeature;
    node.threshold = bestThreshold;
    node.left = buildTree(X, y, leftIndices, depth + 1);
    node.right = buildTree(X, y, rightIndices, depth + 1);
    
    return node;
  }

  private size_t predictSample(Node* node, double[] sample) const {
    if (node is null || node.isLeaf) {
      return node ? node.class_label : 0;
    }
    
    if (sample[node.featureIndex] <= node.threshold) {
      return predictSample(node.left, sample);
    } else {
      return predictSample(node.right, sample);
    }
  }

  private static double informationGain(size_t[] y, size_t[] parent, size_t[] left, size_t[] right) pure {
    double parentEntropy = entropy(y, parent);
    double leftEntropy = entropy(y, left);
    double rightEntropy = entropy(y, right);
    
    double n = parent.length;
    double weightedEntropy = (left.length / n) * leftEntropy + (right.length / n) * rightEntropy;
    
    return parentEntropy - weightedEntropy;
  }

  private static double entropy(size_t[] y, size_t[] indices) pure {
    if (indices.length == 0) return 0.0;
    
    size_t[size_t] counts;
    foreach(idx; indices) {
      counts[y[idx]]++;
    }
    
    double ent = 0.0;
    foreach(count; counts.values) {
      double p = cast(double)count / indices.length;
      if (p > 0.0) {
        ent -= p * log2(p);
      }
    }
    return ent;
  }

  private static size_t majorityClass(size_t[] y, size_t[] indices) pure {
    size_t[size_t] counts;
    foreach(idx; indices) {
      counts[y[idx]]++;
    }
    
    size_t maxCount = 0;
    size_t majorityLabel = 0;
    foreach(label, count; counts) {
      if (count > maxCount) {
        maxCount = count;
        majorityLabel = label;
      }
    }
    return majorityLabel;
  }
}
///
unittest {
  mixin(ShowTest!"Testing DecisionTreeClassifier");

  // Simple dataset
  double[][] X = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]];
  // Class labels
  size_t[] y = [0, 0, 1, 1];
  // Create and fit decision tree
  auto clf = new DecisionTreeClassifier(3, 1);
  // Fit and predict
  clf.fit(X, y);
  // Check predictions on training data
  size_t[] preds = clf.predict(X);
  // Should match original labels
  assert(preds == y, "Predictions should match labels");

  // Test on new samples
  double[][] testX = [[1.2, 1.9], [5.5, 8.5]];
  // Should predict class 0 for first and class 1 for second
  size_t[] testPreds = clf.predict(testX);
  // Check predictions
  assert(testPreds[0] == 0, "First test sample should be class 0");
  // Check predictions
  assert(testPreds[1] == 1, "Second test sample should be class 1");
}


