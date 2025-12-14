"""
Latent Sound Atlas - K-Means Clustering

This script applies K-Means clustering to the 3D PCA coordinates
to group similar sounds together. These clusters will be color-coded
in the 3D visualization.

Input: pca_results/pca_coordinates_3d.npy, metadata.json
Output:
- cluster_assignments.npy (150 x 1 array of cluster IDs)
- kmeans_model.pkl (fitted K-Means object)
- clustering_analysis.json (metrics and statistics)
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
PCA_DIR = PROJECT_ROOT / "pca_results"
PREPARED_DIR = PROJECT_ROOT / "prepared_data"
OUTPUT_DIR = PROJECT_ROOT / "clustering_results"
OUTPUT_DIR.mkdir(exist_ok=True)

N_CLUSTERS = 12  # Number of clusters (10-15 recommended)
RANDOM_STATE = 42

# ============================================================================
# LOAD DATA
# ============================================================================

def load_pca_data():
    """Load PCA coordinates and metadata"""
    print("="*60)
    print("LOADING PCA DATA")
    print("="*60)
    
    # Load 3D coordinates
    coords_path = PCA_DIR / "pca_coordinates_3d.npy"
    if not coords_path.exists():
        raise FileNotFoundError(f"PCA coordinates not found: {coords_path}")
    
    coordinates = np.load(coords_path)
    print(f"✓ Loaded 3D coordinates: {coordinates.shape}")
    
    # Load metadata
    metadata_path = PREPARED_DIR / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded metadata: {metadata['n_samples']} samples")
    
    return coordinates, metadata

# ============================================================================
# OPTIMAL CLUSTER SELECTION
# ============================================================================

def find_optimal_clusters(coordinates, max_clusters=20):
    """
    Find optimal number of clusters using elbow method and silhouette score
    """
    print("\n" + "="*60)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*60)
    
    cluster_range = range(2, max_clusters + 1)
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    
    print("Testing cluster counts from 2 to", max_clusters)
    
    for n in cluster_range:
        kmeans = KMeans(n_clusters=n, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(coordinates)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(coordinates, labels))
        davies_bouldin_scores.append(davies_bouldin_score(coordinates, labels))
        
        if n % 5 == 0:
            print(f"  Tested n={n}: silhouette={silhouette_scores[-1]:.3f}")
    
    # Plot metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Elbow plot
    axes[0].plot(cluster_range, inertias, 'bo-')
    axes[0].set_xlabel('Number of Clusters', fontsize=12)
    axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette score (higher is better)
    axes[1].plot(cluster_range, silhouette_scores, 'go-')
    axes[1].set_xlabel('Number of Clusters', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score (Higher is Better)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Davies-Bouldin score (lower is better)
    axes[2].plot(cluster_range, davies_bouldin_scores, 'ro-')
    axes[2].set_xlabel('Number of Clusters', fontsize=12)
    axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12)
    axes[2].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cluster_optimization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: cluster_optimization.png")
    plt.close()
    
    # Find optimal based on silhouette score
    optimal_n = cluster_range[np.argmax(silhouette_scores)]
    print(f"\n✓ Optimal clusters (by silhouette score): {optimal_n}")
    print(f"  Silhouette score: {max(silhouette_scores):.3f}")
    
    return optimal_n, {
        'cluster_range': list(cluster_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'optimal_clusters': int(optimal_n)
    }

# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

def apply_kmeans(coordinates, n_clusters=12):
    """Apply K-Means clustering to 3D coordinates"""
    print("\n" + "="*60)
    print(f"APPLYING K-MEANS CLUSTERING (n_clusters={n_clusters})")
    print("="*60)
    
    # Initialize and fit K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=20,  # More initializations for stability
        max_iter=500
    )
    
    cluster_labels = kmeans.fit_predict(coordinates)
    
    # Calculate clustering metrics
    silhouette = silhouette_score(coordinates, cluster_labels)
    davies_bouldin = davies_bouldin_score(coordinates, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(coordinates, cluster_labels)
    
    print(f"\n✓ Clustering complete!")
    print(f"  Silhouette Score: {silhouette:.4f} (range: [-1, 1], higher is better)")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    print(f"  Calinski-Harabasz Index: {calinski_harabasz:.4f} (higher is better)")
    
    # Cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\n  Cluster sizes:")
    for cluster_id, count in zip(unique, counts):
        print(f"    Cluster {cluster_id}: {count} sounds")
    
    # Analysis dictionary
    analysis = {
        'n_clusters': n_clusters,
        'n_samples': len(cluster_labels),
        'silhouette_score': float(silhouette),
        'davies_bouldin_index': float(davies_bouldin),
        'calinski_harabasz_index': float(calinski_harabasz),
        'cluster_sizes': {int(cid): int(cnt) for cid, cnt in zip(unique, counts)},
        'cluster_centers': kmeans.cluster_centers_.tolist()
    }
    
    return cluster_labels, kmeans, analysis

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_clusters(coordinates, cluster_labels, metadata, output_dir):
    """Create visualizations of clustering results"""
    print("\n" + "="*60)
    print("GENERATING CLUSTER VISUALIZATIONS")
    print("="*60)
    
    n_clusters = len(np.unique(cluster_labels))
    
    # Generate distinct colors for clusters
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    # 1. 3D Scatter Plot - Colored by Cluster
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            coordinates[mask, 2],
            c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
            alpha=0.7, s=50
        )
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)
    ax.set_title(f'K-Means Clustering Results ({n_clusters} Clusters)', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / 'clusters_3d.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: clusters_3d.png")
    plt.close()
    
    # 2. 2D Projections with Clusters
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PC1 vs PC2
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        axes[0].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            c=[colors[cluster_id]], label=f'C{cluster_id}',
            alpha=0.7, s=40
        )
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('Clusters: PC1 vs PC2')
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # PC1 vs PC3
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        axes[1].scatter(
            coordinates[mask, 0],
            coordinates[mask, 2],
            c=[colors[cluster_id]], label=f'C{cluster_id}',
            alpha=0.7, s=40
        )
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC3')
    axes[1].set_title('Clusters: PC1 vs PC3')
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # PC2 vs PC3
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        axes[2].scatter(
            coordinates[mask, 1],
            coordinates[mask, 2],
            c=[colors[cluster_id]], label=f'C{cluster_id}',
            alpha=0.7, s=40
        )
    axes[2].set_xlabel('PC2')
    axes[2].set_ylabel('PC3')
    axes[2].set_title('Clusters: PC2 vs PC3')
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clusters_2d_projections.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: clusters_2d_projections.png")
    plt.close()
    
    # 3. Cluster Composition by Category
    categories = metadata['categories']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    cluster_category_counts = {}
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_cats = [categories[i] for i, m in enumerate(mask) if m]
        cluster_category_counts[cluster_id] = pd.Series(cluster_cats).value_counts().to_dict()
    
    # Create stacked bar chart
    df_counts = pd.DataFrame(cluster_category_counts).fillna(0).T
    df_counts.plot(kind='bar', stacked=True, ax=ax, color=['#3B82F6', '#10B981', '#F59E0B', '#EF4444'])
    
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Number of Sounds', fontsize=12)
    ax.set_title('Cluster Composition by Sound Category', fontsize=14)
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_composition_by_category.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: cluster_composition_by_category.png")
    plt.close()
    
    # 4. Cluster Composition by Mood
    moods = metadata['moods']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    cluster_mood_counts = {}
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_moods = [moods[i] for i, m in enumerate(mask) if m]
        cluster_mood_counts[cluster_id] = pd.Series(cluster_moods).value_counts().to_dict()
    
    df_mood_counts = pd.DataFrame(cluster_mood_counts).fillna(0).T
    df_mood_counts.plot(kind='bar', stacked=True, ax=ax, color=['#FF6B9D', '#A78BFA', '#EF4444'])
    
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Number of Sounds', fontsize=12)
    ax.set_title('Cluster Composition by Mood', fontsize=14)
    ax.legend(title='Mood', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_composition_by_mood.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: cluster_composition_by_mood.png")
    plt.close()

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_clustering_results(cluster_labels, kmeans, analysis, output_dir):
    """Save clustering results"""
    print("\n" + "="*60)
    print("SAVING CLUSTERING RESULTS")
    print("="*60)
    
    # 1. Save cluster assignments
    np.save(output_dir / 'cluster_assignments.npy', cluster_labels)
    print(f"✓ Saved: cluster_assignments.npy")
    
    # 2. Save K-Means model
    with open(output_dir / 'kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"✓ Saved: kmeans_model.pkl")
    
    # 3. Save analysis
    with open(output_dir / 'clustering_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Saved: clustering_analysis.json")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main clustering pipeline"""
    print("="*60)
    print("LATENT SOUND ATLAS - K-MEANS CLUSTERING")
    print("="*60)
    
    coordinates, metadata = load_pca_data()
    
    cluster_labels, kmeans, analysis = apply_kmeans(coordinates, n_clusters=N_CLUSTERS)

    visualize_clusters(coordinates, cluster_labels, metadata, OUTPUT_DIR)

    save_clustering_results(cluster_labels, kmeans, analysis, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✓ K-MEANS CLUSTERING COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - Cluster assignments: {len(cluster_labels)} sounds")
    print(f"  - Number of clusters: {N_CLUSTERS}")
    print(f"  - Silhouette score: {analysis['silhouette_score']:.4f}")
    print(f"  - Visualizations: 4 PNG files")

if __name__ == "__main__":
    main()