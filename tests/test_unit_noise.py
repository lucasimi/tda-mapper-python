"""Tests for noise point handling in clustering."""

import numpy as np
from sklearn.cluster import DBSCAN
from tdamapper.core import (
    NoiseHandlingClustering,
    TrivialCover,
    mapper_connected_components,
)


def test_noise_handling_clustering():
    # Create a simple dataset with obvious noise points
    X = np.array([
        [0, 0],  # Cluster 1
        [0.1, 0.1],  # Cluster 1
        [5, 5],  # Noise point
        [1, 1],  # Cluster 2
        [1.1, 0.9],  # Cluster 2
        [10, 10],  # Noise point
    ])

    # Base clustering with DBSCAN (eps=0.3 will make points far apart noise)
    base_clustering = DBSCAN(eps=0.3, min_samples=2)  # min_samples=2 to ensure small clusters are valid
    # Debug: Print raw DBSCAN labels
    debug_labels = base_clustering.fit(X).labels_
    print(f"\nDebug - Raw DBSCAN labels: {debug_labels}")
    
    # Test invalid noise_handling parameter
    try:
        NoiseHandlingClustering(clustering=base_clustering, noise_handling='invalid')
        assert False, "Should raise ValueError for invalid noise_handling"
    except ValueError as e:
        assert "noise_handling must be one of" in str(e)
    
    # Test 'drop' mode
    clustering_drop = NoiseHandlingClustering(
        clustering=base_clustering,
        noise_handling='drop'
    )
    clustering_drop.fit(X)
    assert -1 in clustering_drop.labels_, "Noise points should be kept as -1"
    
    # Test 'group' mode
    clustering_group = NoiseHandlingClustering(
        clustering=base_clustering,
        noise_handling='group'
    )
    clustering_group.fit(X)
    assert -1 not in clustering_group.labels_, "No points should be marked as noise"
    noise_points = np.where(clustering_group.labels_ == max(clustering_group.labels_))[0]
    assert len(noise_points) == 2, "Should have 2 points in noise cluster"
    assert 2 in noise_points and 5 in noise_points, "Points [5,5] and [10,10] should be noise"
    
    # Test 'singleton' mode (default)
    clustering_singleton = NoiseHandlingClustering(
        clustering=base_clustering
    )
    clustering_singleton.fit(X)
    assert -1 not in clustering_singleton.labels_, "No points should be marked as noise"
    # Each noise point should have its own unique label
    noise_labels = clustering_singleton.labels_[[2, 5]]  # labels for [5,5] and [10,10]
    assert len(set(noise_labels)) == 2, "Each noise point should have unique label"
    # Verify exact number of clusters (2 original clusters + 2 singleton noise clusters)
    assert len(set(clustering_singleton.labels_)) == 4, "Should have exactly 4 clusters (2 original + 2 noise)"


def test_mapper_with_noise_handling():
    # Create a dataset with noise points
    X = np.array([
        [0, 0], [0.1, 0.1],  # Cluster 1
        [5, 5],  # Noise point
        [1, 1], [1.1, 0.9],  # Cluster 2
        [10, 10],  # Noise point
    ])
    
    # Test with default noise handling (drop)
    base_clustering = DBSCAN(eps=0.3)
    labels = mapper_connected_components(
        X, X,  # Use X as both data and lens
        TrivialCover(),
        base_clustering
    )
    assert -1 in labels, "Noise points should be kept by default"
    
    # Test with custom noise handling
    noise_handler = NoiseHandlingClustering(
        clustering=base_clustering,
        noise_handling='singleton'
    )
    labels = mapper_connected_components(
        X, X,
        TrivialCover(),
        noise_handler
    )
    assert -1 not in labels, "No points should be marked as noise"
    assert len(set(labels)) >= 4, "Should have at least 4 components (2 clusters + 2 noise)"