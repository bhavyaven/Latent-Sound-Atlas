"""
Latent Sound Atlas - Week 2, Task 2.1
PCA Dimensionality Reduction

This script applies PCA to reduce the 47-dimensional feature space
to 3D coordinates (X, Y, Z) for visualization.

Input: prepared_data/features_scaled.npy, metadata.json
Output: 
- pca_coordinates_3d.npy (150 x 3 array)
- pca_model.pkl (fitted PCA object)
- pca_analysis.json (variance explained, etc.)
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / "prepared_data"
OUTPUT_DIR = PROJECT_ROOT / "pca_results"
OUTPUT_DIR.mkdir(exist_ok=True)

N_COMPONENTS = 3  # X, Y, Z coordinates

# ============================================================================
# LOAD PREPARED DATA
# ============================================================================

def load_prepared_data():
    """Load scaled features and metadata"""
    print("="*60)
    print("LOADING PREPARED DATA")
    print("="*60)
    
    # Load scaled features
    features_path = INPUT_DIR / "features_scaled.npy"
    if not features_path.exists():
        raise FileNotFoundError(f"Scaled features not found: {features_path}")
    
    features = np.load(features_path)
    print(f"✓ Loaded features: {features.shape}")
    
    # Load metadata
    metadata_path = INPUT_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded metadata: {metadata['n_samples']} samples")
    print(f"  Categories: {set(metadata['categories'])}")
    print(f"  Moods: {set(metadata['moods'])}")
    
    return features, metadata

# ============================================================================
# PCA DIMENSIONALITY REDUCTION
# ============================================================================

def apply_pca(features, n_components=3):
    """Apply PCA to reduce features to 3D"""
    print("\n" + "="*60)
    print(f"APPLYING PCA (reducing to {n_components}D)")
    print("="*60)
    
    # Initialize PCA
    pca = PCA(n_components=n_components, random_state=42)
    
    # Fit and transform
    print(f"Input shape: {features.shape}")
    coordinates_3d = pca.fit_transform(features)
    print(f"Output shape: {coordinates_3d.shape}")
    
    # Variance explained
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    
    print(f"\n✓ PCA complete!")
    print(f"  PC1 variance: {variance_explained[0]:.4f} ({variance_explained[0]*100:.2f}%)")
    print(f"  PC2 variance: {variance_explained[1]:.4f} ({variance_explained[1]*100:.2f}%)")
    print(f"  PC3 variance: {variance_explained[2]:.4f} ({variance_explained[2]*100:.2f}%)")
    print(f"  Total variance explained: {cumulative_variance[-1]:.4f} ({cumulative_variance[-1]*100:.2f}%)")
    
    # Analysis dictionary
    analysis = {
        'n_components': n_components,
        'n_samples': features.shape[0],
        'n_features_original': features.shape[1],
        'variance_explained': variance_explained.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'total_variance_explained': float(cumulative_variance[-1]),
        'principal_components': {
            'PC1': float(variance_explained[0]),
            'PC2': float(variance_explained[1]),
            'PC3': float(variance_explained[2])
        }
    }
    
    return coordinates_3d, pca, analysis

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_pca_results(coordinates_3d, metadata, output_dir):
    """Create visualizations of PCA results"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Extract metadata for coloring
    categories = metadata['categories']
    moods = metadata['moods']
    synths = metadata['synths']
    
    # 1. 3D Scatter Plot - Colored by Category
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    category_colors = {'mus': 'blue', 'fx': 'green', 'amb': 'orange', 'sfx': 'red'}
    for category, color in category_colors.items():
        mask = [c == category for c in categories]
        ax.scatter(
            coordinates_3d[mask, 0],
            coordinates_3d[mask, 1],
            coordinates_3d[mask, 2],
            c=color, label=category, alpha=0.6, s=50
        )
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)
    ax.set_title('PCA 3D Projection - Colored by Category', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_3d_by_category.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: pca_3d_by_category.png")
    plt.close()
    
    # 2. 3D Scatter Plot - Colored by Mood
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    mood_colors = {'cozy': '#FF6B9D', 'mystic': '#A78BFA', 'tense': '#EF4444'}
    for mood, color in mood_colors.items():
        mask = [m == mood for m in moods]
        ax.scatter(
            coordinates_3d[mask, 0],
            coordinates_3d[mask, 1],
            coordinates_3d[mask, 2],
            c=color, label=mood, alpha=0.6, s=50
        )
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)
    ax.set_title('PCA 3D Projection - Colored by Mood', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_3d_by_mood.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: pca_3d_by_mood.png")
    plt.close()
    
    # 3. 2D Projections (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PC1 vs PC2
    for category, color in category_colors.items():
        mask = [c == category for c in categories]
        axes[0].scatter(
            coordinates_3d[mask, 0],
            coordinates_3d[mask, 1],
            c=color, label=category, alpha=0.6, s=30
        )
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('PC1 vs PC2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PC1 vs PC3
    for category, color in category_colors.items():
        mask = [c == category for c in categories]
        axes[1].scatter(
            coordinates_3d[mask, 0],
            coordinates_3d[mask, 2],
            c=color, label=category, alpha=0.6, s=30
        )
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC3')
    axes[1].set_title('PC1 vs PC3')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # PC2 vs PC3
    for category, color in category_colors.items():
        mask = [c == category for c in categories]
        axes[2].scatter(
            coordinates_3d[mask, 1],
            coordinates_3d[mask, 2],
            c=color, label=category, alpha=0.6, s=30
        )
    axes[2].set_xlabel('PC2')
    axes[2].set_ylabel('PC3')
    axes[2].set_title('PC2 vs PC3')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_2d_projections.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: pca_2d_projections.png")
    plt.close()
    
    # 4. Coordinate Distribution Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(coordinates_3d[:, 0], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('PC1 Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('PC1 Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(coordinates_3d[:, 1], bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('PC2 Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('PC2 Distribution')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(coordinates_3d[:, 2], bins=30, edgecolor='black', alpha=0.7, color='mediumseagreen')
    axes[2].set_xlabel('PC3 Value')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('PC3 Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_coordinate_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: pca_coordinate_distributions.png")
    plt.close()

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_pca_results(coordinates_3d, pca, analysis, metadata, output_dir):
    """Save PCA coordinates and model"""
    print("\n" + "="*60)
    print("SAVING PCA RESULTS")
    print("="*60)
    
    # 1. Save 3D coordinates as NumPy array
    np.save(output_dir / 'pca_coordinates_3d.npy', coordinates_3d)
    print(f"✓ Saved: pca_coordinates_3d.npy")
    print(f"  Shape: {coordinates_3d.shape}")
    
    # 2. Save PCA model
    with open(output_dir / 'pca_model.pkl', 'wb') as f:
        pickle.dump(pca, f)
    print(f"✓ Saved: pca_model.pkl")
    
    # 3. Save analysis results
    with open(output_dir / 'pca_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Saved: pca_analysis.json")
    
    # 4. Save coordinates with metadata as CSV (for inspection)
    df = pd.DataFrame({
        'filename': metadata['filenames'],
        'category': metadata['categories'],
        'synth': metadata['synths'],
        'type': metadata['types'],
        'mood': metadata['moods'],
        'index': metadata['indices'],
        'x': coordinates_3d[:, 0],
        'y': coordinates_3d[:, 1],
        'z': coordinates_3d[:, 2]
    })
    
    df.to_csv(output_dir / 'pca_coordinates_with_metadata.csv', index=False)
    print(f"✓ Saved: pca_coordinates_with_metadata.csv")
    
    # 5. Save coordinate statistics
    coord_stats = {
        'x': {
            'min': float(coordinates_3d[:, 0].min()),
            'max': float(coordinates_3d[:, 0].max()),
            'mean': float(coordinates_3d[:, 0].mean()),
            'std': float(coordinates_3d[:, 0].std())
        },
        'y': {
            'min': float(coordinates_3d[:, 1].min()),
            'max': float(coordinates_3d[:, 1].max()),
            'mean': float(coordinates_3d[:, 1].mean()),
            'std': float(coordinates_3d[:, 1].std())
        },
        'z': {
            'min': float(coordinates_3d[:, 2].min()),
            'max': float(coordinates_3d[:, 2].max()),
            'mean': float(coordinates_3d[:, 2].mean()),
            'std': float(coordinates_3d[:, 2].std())
        }
    }
    
    with open(output_dir / 'coordinate_statistics.json', 'w') as f:
        json.dump(coord_stats, f, indent=2)
    print(f"✓ Saved: coordinate_statistics.json")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main PCA pipeline"""
    print("="*60)
    print("LATENT SOUND ATLAS - PCA DIMENSIONALITY REDUCTION")
    print("="*60)
    
    # 1. Load data
    features, metadata = load_prepared_data()
    
    # 2. Apply PCA
    coordinates_3d, pca, analysis = apply_pca(features, n_components=N_COMPONENTS)
    
    # 3. Visualize
    visualize_pca_results(coordinates_3d, metadata, OUTPUT_DIR)
    
    # 4. Save results
    save_pca_results(coordinates_3d, pca, analysis, metadata, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✓ PCA DIMENSIONALITY REDUCTION COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - 3D coordinates: {coordinates_3d.shape}")
    print(f"  - Variance explained: {analysis['total_variance_explained']*100:.2f}%")
    print(f"  - Visualizations: 4 PNG files")
    print("\n✓ Ready for Task 2.2: Clustering")
    print("   Run: python clustering.py")

if __name__ == "__main__":
    main()