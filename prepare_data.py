"""
Latent Sound Atlas - Week 1, Task 1.3
Data Preparation and Scaling Script

This script loads the FluCoMa-generated features, cleans the data,
and prepares scaled NumPy arrays for dimensionality reduction.

Input: master_features.csv
Output: 
- features_scaled.npy (scaled feature matrix)
- metadata.json (labels and metadata)
- scaler.pkl (fitted scaler for later use)
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
FEATURE_DIR = PROJECT_ROOT / "feature_data"
OUTPUT_DIR = PROJECT_ROOT / "prepared_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Scaling method: 'standard', 'minmax', or 'robust'
SCALING_METHOD = 'standard'

# Feature selection strategy
FEATURE_COLUMNS_TO_EXCLUDE = ['filename', 'category', 'synth', 'type', 'mood', 'index']

# ============================================================================
# DATA LOADING
# ============================================================================

def load_features():
    """Load the master features CSV"""
    csv_path = FEATURE_DIR / "master_features.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")
    
    return df

# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_data(df):
    """Clean and validate feature data"""
    print("\n=== DATA CLEANING ===")
    
    # Separate metadata from features
    metadata_cols = FEATURE_COLUMNS_TO_EXCLUDE
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    metadata = df[metadata_cols].copy()
    features = df[feature_cols].copy()
    
    print(f"Metadata columns: {len(metadata_cols)}")
    print(f"Feature columns: {len(feature_cols)}")
    
    # Ensure all feature columns are numeric
    print("\n=== CHECKING DATA TYPES ===")
    non_numeric_cols = []
    for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        print(f"⚠ Found {len(non_numeric_cols)} non-numeric columns:")
        for col in non_numeric_cols[:5]:
            print(f"  - {col}: {features[col].dtype}")
        print("Converting to numeric (coercing errors to NaN)...")
        for col in non_numeric_cols:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        print("✓ Converted to numeric")
    else:
        print("✓ All feature columns are numeric")
    
    # Check for missing values
    missing_counts = features.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\n⚠ Found {missing_counts.sum()} missing values")
        print("Missing value counts per column:")
        print(missing_counts[missing_counts > 0])
        
        # Impute missing values with median
        imputer = SimpleImputer(strategy='median')
        features_imputed = pd.DataFrame(
            imputer.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        print("✓ Imputed missing values with median")
        features = features_imputed
    else:
        print("✓ No missing values found")
    
    # Check for infinite values
    inf_mask = np.isinf(features.values.astype(float))
    if inf_mask.any():
        print(f"\n⚠ Found {inf_mask.sum()} infinite values")
        # Replace inf with max finite value per column
        for col in features.columns:
            col_data = features[col]
            if np.isinf(col_data).any():
                max_finite = col_data[np.isfinite(col_data)].max()
                min_finite = col_data[np.isfinite(col_data)].min()
                features[col] = col_data.replace([np.inf], max_finite)
                features[col] = features[col].replace([-np.inf], min_finite)
        print("✓ Replaced infinite values")
    else:
        print("✓ No infinite values found")
    
    # Check for constant features (zero variance)
    zero_var_cols = features.columns[features.var() == 0]
    if len(zero_var_cols) > 0:
        print(f"\n⚠ Found {len(zero_var_cols)} zero-variance features")
        print(f"Removing: {list(zero_var_cols)}")
        features = features.drop(columns=zero_var_cols)
    else:
        print("✓ No zero-variance features")
    
    # Check for highly correlated features (optional filtering)
    correlation_matrix = features.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [
        col for col in upper_triangle.columns 
        if any(upper_triangle[col] > 0.95)
    ]
    
    if len(high_corr_features) > 0:
        print(f"\n⚠ Found {len(high_corr_features)} highly correlated features (>0.95)")
        print(f"Consider removing: {high_corr_features[:5]}...")
    else:
        print("✓ No highly correlated features (>0.95)")
    
    return metadata, features

# ============================================================================
# FEATURE SCALING
# ============================================================================

def scale_features(features, method='standard'):
    """Scale features using specified method"""
    print(f"\n=== FEATURE SCALING (method: {method}) ===")
    
    if method == 'standard':
        # Standardization: zero mean, unit variance
        scaler = StandardScaler()
        print("Using StandardScaler (z-score normalization)")
    elif method == 'minmax':
        # Min-Max scaling: [0, 1] range
        scaler = MinMaxScaler()
        print("Using MinMaxScaler (range [0, 1])")
    elif method == 'robust':
        # Robust scaling: median and IQR (good for outliers)
        scaler = RobustScaler()
        print("Using RobustScaler (median and IQR)")
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit and transform
    features_scaled = scaler.fit_transform(features.values)
    
    print(f"✓ Scaled feature matrix shape: {features_scaled.shape}")
    print(f"  Mean: {features_scaled.mean():.6f}")
    print(f"  Std: {features_scaled.std():.6f}")
    print(f"  Min: {features_scaled.min():.6f}")
    print(f"  Max: {features_scaled.max():.6f}")
    
    return features_scaled, scaler

# ============================================================================
# DATA VISUALIZATION
# ============================================================================

def visualize_distributions(features, features_scaled, output_dir):
    """Create visualization of feature distributions before/after scaling"""
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # Select a few example features to plot
    example_features = features.columns[:6]
    example_indices = [features.columns.get_loc(col) for col in example_features]
    
    fig, axes = plt.subplots(2, len(example_features), figsize=(18, 8))
    fig.suptitle('Feature Distributions: Before and After Scaling', fontsize=16)
    
    for i, (col, idx) in enumerate(zip(example_features, example_indices)):
        # Before scaling
        axes[0, i].hist(features[col], bins=30, edgecolor='black', alpha=0.7)
        axes[0, i].set_title(f'{col}\n(Original)', fontsize=10)
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')
        
        # After scaling
        axes[1, i].hist(features_scaled[:, idx], bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1, i].set_title('(Scaled)', fontsize=10)
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'feature_distributions.png'}")
    plt.close()
    
    # Correlation heatmap (on scaled features, first 20 features only)
    plt.figure(figsize=(14, 12))
    corr_subset = pd.DataFrame(features_scaled[:, :20]).corr()
    sns.heatmap(corr_subset, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix (First 20 Features)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlation.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'feature_correlation.png'}")
    plt.close()

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_prepared_data(features_scaled, metadata, scaler, feature_names, output_dir):
    """Save all prepared data and metadata"""
    print("\n=== SAVING PREPARED DATA ===")
    
    # Save scaled features as NumPy array
    np.save(output_dir / 'features_scaled.npy', features_scaled)
    print(f"✓ Saved: {output_dir / 'features_scaled.npy'}")
    print(f"  Shape: {features_scaled.shape}")
    
    # Save metadata as JSON
    metadata_dict = {
        'filenames': metadata['filename'].tolist(),
        'categories': metadata['category'].tolist(),
        'synths': metadata['synth'].tolist(),
        'types': metadata['type'].tolist(),
        'moods': metadata['mood'].tolist(),
        'indices': metadata['index'].tolist(),
        'feature_names': feature_names,
        'n_samples': len(metadata),
        'n_features': features_scaled.shape[1],
        'scaling_method': SCALING_METHOD
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"✓ Saved: {output_dir / 'metadata.json'}")
    
    # Save scaler for future use
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved: {output_dir / 'scaler.pkl'}")
    
    # Save feature names separately
    with open(output_dir / 'feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print(f"✓ Saved: {output_dir / 'feature_names.txt'}")
    
    # Create summary statistics
    summary = {
        'n_samples': int(features_scaled.shape[0]),
        'n_features': int(features_scaled.shape[1]),
        'scaling_method': SCALING_METHOD,
        'feature_statistics': {
            'mean': float(features_scaled.mean()),
            'std': float(features_scaled.std()),
            'min': float(features_scaled.min()),
            'max': float(features_scaled.max())
        },
        'category_distribution': metadata['category'].value_counts().to_dict(),
        'synth_distribution': metadata['synth'].value_counts().to_dict(),
        'mood_distribution': metadata['mood'].value_counts().to_dict(),
        'type_distribution': metadata['type'].value_counts().to_dict()
    }
    
    with open(output_dir / 'data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: {output_dir / 'data_summary.json'}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main data preparation pipeline"""
    print("="*60)
    print("LATENT SOUND ATLAS - DATA PREPARATION")
    print("="*60)
    
    # 1. Load features
    df = load_features()
    
    # 2. Clean data
    metadata, features = clean_data(df)
    
    # 3. Scale features
    features_scaled, scaler = scale_features(features, method=SCALING_METHOD)
    
    # 4. Visualize
    visualize_distributions(features, features_scaled, OUTPUT_DIR)
    
    # 5. Save everything
    save_prepared_data(
        features_scaled, 
        metadata, 
        scaler, 
        features.columns.tolist(),
        OUTPUT_DIR
    )
    
    print("\n" + "="*60)
    print("✓ DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - features_scaled.npy: {features_scaled.shape}")
    print(f"  - metadata.json: {len(metadata)} samples")
    print(f"  - scaler.pkl: {SCALING_METHOD} scaler")
    print(f"  - Visualizations: 2 PNG files")
    print("\nReady for dimensionality reduction (Week 2)!")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_prepared_data():
    """Utility function to load prepared data in other scripts"""
    features = np.load(OUTPUT_DIR / 'features_scaled.npy')
    
    with open(OUTPUT_DIR / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    with open(OUTPUT_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    return features, metadata, scaler

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()