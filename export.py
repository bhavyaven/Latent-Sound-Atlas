"""
Latent Sound Atlas - Data Export for OpenGL/3D Visualization

This script combines PCA coordinates, cluster assignments, and metadata
into formats optimized for 3D rendering (JSON, CSV).

Input: 
- pca_results/pca_coordinates_3d.npy
- clustering_results/cluster_assignments.npy
- prepared_data/metadata.json

Output:
- sound_map_data.json (complete dataset for OpenGL)
- sound_map_data.csv (human-readable format)
- cluster_colors.json (color palette for clusters)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import colorsys

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
PCA_DIR = PROJECT_ROOT / "pca_results"
CLUSTERING_DIR = PROJECT_ROOT / "clustering_results"
PREPARED_DIR = PROJECT_ROOT / "prepared_data"
OUTPUT_DIR = PROJECT_ROOT / "opengl_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# COLOR GENERATION
# ============================================================================

def generate_cluster_colors(n_clusters):
    """Generate distinct colors for each cluster"""
    colors = []
    
    for i in range(n_clusters):
        # Use HSV color space for distinct hues
        hue = i / n_clusters
        saturation = 0.7
        value = 0.9
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        colors.append({
            'cluster_id': i,
            'rgb': [float(rgb[0]), float(rgb[1]), float(rgb[2])],
            'rgba': [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0],
            'hex': '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
        })
    
    return colors

# ============================================================================
# LOAD ALL DATA
# ============================================================================

def load_all_data():
    """Load PCA coordinates, clusters, and metadata"""
    print("="*60)
    print("LOADING ALL DATA FOR EXPORT")
    print("="*60)
    
    # 1. Load 3D coordinates
    coords_path = PCA_DIR / "pca_coordinates_3d.npy"
    if not coords_path.exists():
        raise FileNotFoundError(f"PCA coordinates not found: {coords_path}")
    coordinates = np.load(coords_path)
    print(f"✓ Loaded coordinates: {coordinates.shape}")
    
    # 2. Load cluster assignments
    clusters_path = CLUSTERING_DIR / "cluster_assignments.npy"
    if not clusters_path.exists():
        raise FileNotFoundError(f"Cluster assignments not found: {clusters_path}")
    cluster_labels = np.load(clusters_path)
    print(f"✓ Loaded clusters: {len(cluster_labels)} assignments")
    
    # 3. Load metadata
    metadata_path = PREPARED_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Loaded metadata: {metadata['n_samples']} samples")
    
    # 4. Load PCA analysis (for variance info)
    pca_analysis_path = PCA_DIR / "pca_analysis.json"
    with open(pca_analysis_path, 'r') as f:
        pca_analysis = json.load(f)
    
    # 5. Load clustering analysis
    cluster_analysis_path = CLUSTERING_DIR / "clustering_analysis.json"
    with open(cluster_analysis_path, 'r') as f:
        cluster_analysis = json.load(f)
    
    return coordinates, cluster_labels, metadata, pca_analysis, cluster_analysis

# ============================================================================
# NORMALIZE COORDINATES
# ============================================================================

def normalize_coordinates(coordinates, method='minmax', scale_factor=100.0):
    """
    Normalize coordinates to a reasonable range for OpenGL
    
    Methods:
    - 'minmax': Scale to [0, scale_factor] range
    - 'centered': Center at origin, scale to [-scale_factor/2, scale_factor/2]
    - 'standard': Standardize to mean=0, std=scale_factor
    """
    print("\n" + "="*60)
    print(f"NORMALIZING COORDINATES (method: {method})")
    print("="*60)
    
    coords_normalized = coordinates.copy()
    
    if method == 'minmax':
        # Scale to [0, scale_factor]
        for i in range(3):
            min_val = coordinates[:, i].min()
            max_val = coordinates[:, i].max()
            coords_normalized[:, i] = (coordinates[:, i] - min_val) / (max_val - min_val) * scale_factor
        print(f"✓ Scaled to range [0, {scale_factor}]")
    
    elif method == 'centered':
        # Center at origin and scale
        for i in range(3):
            coords_normalized[:, i] = coordinates[:, i] - coordinates[:, i].mean()
            max_abs = np.abs(coords_normalized[:, i]).max()
            coords_normalized[:, i] = (coords_normalized[:, i] / max_abs) * (scale_factor / 2)
        print(f"✓ Centered at origin, range [{-scale_factor/2}, {scale_factor/2}]")
    
    elif method == 'standard':
        # Standardize and scale
        for i in range(3):
            mean = coordinates[:, i].mean()
            std = coordinates[:, i].std()
            coords_normalized[:, i] = ((coordinates[:, i] - mean) / std) * scale_factor
        print(f"✓ Standardized with scale factor {scale_factor}")
    
    print(f"  X range: [{coords_normalized[:, 0].min():.2f}, {coords_normalized[:, 0].max():.2f}]")
    print(f"  Y range: [{coords_normalized[:, 1].min():.2f}, {coords_normalized[:, 1].max():.2f}]")
    print(f"  Z range: [{coords_normalized[:, 2].min():.2f}, {coords_normalized[:, 2].max():.2f}]")
    
    return coords_normalized

# ============================================================================
# CREATE MASTER DATASET
# ============================================================================

def create_master_dataset(coordinates, cluster_labels, metadata, pca_analysis, cluster_analysis):
    """Combine all data into a master dataset"""
    print("\n" + "="*60)
    print("CREATING MASTER DATASET")
    print("="*60)
    
    n_samples = len(metadata['filenames'])
    n_clusters = cluster_analysis['n_clusters']
    
    # Generate cluster colors
    cluster_colors = generate_cluster_colors(n_clusters)
    
    # Create point-by-point dataset
    points = []
    for i in range(n_samples):
        cluster_id = int(cluster_labels[i])
        point = {
            'id': i,
            'filename': metadata['filenames'][i],
            'category': metadata['categories'][i],
            'synth': metadata['synths'][i],
            'type': metadata['types'][i],
            'mood': metadata['moods'][i],
            'index': metadata['indices'][i],
            'coordinates': {
                'x': float(coordinates[i, 0]),
                'y': float(coordinates[i, 1]),
                'z': float(coordinates[i, 2])
            },
            'cluster_id': cluster_id,
            'color': {
                'rgb': cluster_colors[cluster_id]['rgb'],
                'rgba': cluster_colors[cluster_id]['rgba'],
                'hex': cluster_colors[cluster_id]['hex']
            }
        }
        points.append(point)
    
    # Create master JSON structure
    master_data = {
        'metadata': {
            'project': 'Latent Sound Atlas',
            'version': '1.0',
            'n_samples': n_samples,
            'n_clusters': n_clusters,
            'generation_date': pd.Timestamp.now().isoformat()
        },
        'pca_info': {
            'n_components': 3,
            'variance_explained': pca_analysis['variance_explained'],
            'total_variance': pca_analysis['total_variance_explained']
        },
        'clustering_info': {
            'algorithm': 'K-Means',
            'n_clusters': n_clusters,
            'silhouette_score': cluster_analysis['silhouette_score'],
            'cluster_sizes': cluster_analysis['cluster_sizes']
        },
        'cluster_colors': cluster_colors,
        'points': points,
        'bounds': {
            'x': {'min': float(coordinates[:, 0].min()), 'max': float(coordinates[:, 0].max())},
            'y': {'min': float(coordinates[:, 1].min()), 'max': float(coordinates[:, 1].max())},
            'z': {'min': float(coordinates[:, 2].min()), 'max': float(coordinates[:, 2].max())}
        }
    }
    
    print(f"✓ Created master dataset:")
    print(f"  {n_samples} points")
    print(f"  {n_clusters} clusters")
    print(f"  4 categories: mus, fx, amb, sfx")
    print(f"  3 moods: cozy, mystic, tense")
    
    return master_data

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_json(master_data, output_dir):
    """Export master dataset as JSON"""
    print("\n" + "="*60)
    print("EXPORTING JSON FORMAT")
    print("="*60)
    
    # Full JSON
    json_path = output_dir / 'sound_map_data.json'
    with open(json_path, 'w') as f:
        json.dump(master_data, f, indent=2)
    print(f"✓ Saved: sound_map_data.json")
    print(f"  Size: {json_path.stat().st_size / 1024:.2f} KB")
    
    # Compact JSON (no indentation)
    json_compact_path = output_dir / 'sound_map_data_compact.json'
    with open(json_compact_path, 'w') as f:
        json.dump(master_data, f, separators=(',', ':'))
    print(f"✓ Saved: sound_map_data_compact.json")
    print(f"  Size: {json_compact_path.stat().st_size / 1024:.2f} KB")

def export_csv(master_data, output_dir):
    """Export master dataset as CSV"""
    print("\n" + "="*60)
    print("EXPORTING CSV FORMAT")
    print("="*60)
    
    # Flatten points data for CSV
    rows = []
    for point in master_data['points']:
        row = {
            'id': point['id'],
            'filename': point['filename'],
            'category': point['category'],
            'synth': point['synth'],
            'type': point['type'],
            'mood': point['mood'],
            'index': point['index'],
            'x': point['coordinates']['x'],
            'y': point['coordinates']['y'],
            'z': point['coordinates']['z'],
            'cluster_id': point['cluster_id'],
            'color_hex': point['color']['hex']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = output_dir / 'sound_map_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: sound_map_data.csv")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

def export_cluster_info(master_data, output_dir):
    """Export cluster-specific information"""
    print("\n" + "="*60)
    print("EXPORTING CLUSTER INFO")
    print("="*60)
    
    # Cluster colors only
    colors_path = output_dir / 'cluster_colors.json'
    with open(colors_path, 'w') as f:
        json.dump(master_data['cluster_colors'], f, indent=2)
    print(f"✓ Saved: cluster_colors.json")
    
    # Cluster statistics
    cluster_stats = []
    for cluster in master_data['cluster_colors']:
        cluster_id = cluster['cluster_id']
        cluster_points = [p for p in master_data['points'] if p['cluster_id'] == cluster_id]
        
        # Calculate cluster center
        coords = np.array([[p['coordinates']['x'], p['coordinates']['y'], p['coordinates']['z']] 
                           for p in cluster_points])
        center = coords.mean(axis=0)
        
        # Count by category
        categories = {}
        for p in cluster_points:
            cat = p['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        stat = {
            'cluster_id': cluster_id,
            'size': len(cluster_points),
            'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
            'color': cluster['hex'],
            'composition': categories
        }
        cluster_stats.append(stat)
    
    stats_path = output_dir / 'cluster_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(cluster_stats, f, indent=2)
    print(f"✓ Saved: cluster_statistics.json")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main export pipeline"""
    print("="*60)
    print("LATENT SOUND ATLAS - DATA EXPORT FOR OPENGL")
    print("="*60)
    
    # 1. Load all data
    coordinates, cluster_labels, metadata, pca_analysis, cluster_analysis = load_all_data()
    
    # 2. Normalize coordinates for OpenGL
    coordinates_normalized = normalize_coordinates(coordinates, method='centered', scale_factor=100.0)
    
    # 3. Create master dataset
    master_data = create_master_dataset(
        coordinates_normalized, 
        cluster_labels, 
        metadata, 
        pca_analysis, 
        cluster_analysis
    )
    
    # 4. Export in multiple formats
    export_json(master_data, OUTPUT_DIR)
    export_csv(master_data, OUTPUT_DIR)
    export_cluster_info(master_data, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✓ DATA EXPORT COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - sound_map_data.json (main data file)")
    print(f"  - sound_map_data.csv (human-readable)")
    print(f"  - cluster_colors.json (color palette)")
    print(f"  - cluster_statistics.json (cluster info)")

if __name__ == "__main__":
    main()