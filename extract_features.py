"""
Latent Sound Atlas - Feature Extraction using Librosa (FluCoMa Alternative)

This script processes 150 audio files and extracts acoustic features using librosa.
Features extracted:
- MFCCs (13 coefficients)
- Spectral Centroid
- Spectral Contrast
- Spectral Rolloff
- Zero Crossing Rate
- RMS Energy

Output: master_features.csv with 150 rows
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import librosa
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
AUDIO_DIR = PROJECT_ROOT / "sound_assets"
OUTPUT_DIR = PROJECT_ROOT / "feature_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Feature extraction parameters
SR = 44100  # Sample rate
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_features(audio_file):
    """Extract all audio features from a single file"""
    try:
        # Load audio
        y, sr = librosa.load(str(audio_file), sr=SR)
        
        features = {}
        
        # 1. MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(N_MFCC):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # 2. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        features['centroid_mean'] = np.mean(centroid)
        features['centroid_std'] = np.std(centroid)
        features['centroid_min'] = np.min(centroid)
        features['centroid_max'] = np.max(centroid)
        
        # 3. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features['contrast_mean'] = np.mean(contrast)
        features['contrast_std'] = np.std(contrast)
        
        # 4. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_std'] = np.std(rolloff)
        
        # 5. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        features['flatness_mean'] = np.mean(flatness)
        features['flatness_std'] = np.std(flatness)
        
        # 6. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 7. RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_min'] = np.min(rms)
        features['rms_max'] = np.max(rms)
        
        # 8. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        features['bandwidth_mean'] = np.mean(bandwidth)
        features['bandwidth_std'] = np.std(bandwidth)
        
        # 9. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # 10. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        return features
        
    except Exception as e:
        print(f"Error processing {audio_file.name}: {e}")
        return None

def parse_filename_metadata(filename):
    """
    Extract metadata from filename structure
    Format: category_synthtype_moodindex_.wav
    Example: amb_brooktexture_cozy000_.wav
    
    Pattern breakdown:
    - category: mus, fx, amb, sfx (3 chars)
    - synth+type: compound word (e.g., "brooktexture", "harpsingle", "flutemelody")
    - mood+index: mood name + 3-digit number (e.g., "cozy000", "mystic042")
    """
    stem = filename.stem  # Remove .wav extension
    
    # Remove trailing underscore if present
    if stem.endswith('_'):
        stem = stem[:-1]
    
    # Split by underscores
    parts = stem.split('_')
    
    # Extract category (first part)
    category = parts[0] if len(parts) > 0 else 'unknown'
    
    # Extract synth+type compound (second part)
    synthtype = parts[1] if len(parts) > 1 else 'unknown'
    
    # Parse synth and type from compound word
    # Known synth names (ordered by length to match longest first)
    synth_names = ['sparkle', 'ocarina', 'leaves', 'whoosh', 'brook', 'rain', 
                   'fire', 'bird', 'harp', 'flute', 'chime', 'owl', 'pad']
    
    synth = 'unknown'
    type_val = 'unknown'
    
    for known_synth in synth_names:
        if synthtype.startswith(known_synth):
            synth = known_synth
            type_val = synthtype[len(known_synth):]  # Rest is the type
            break
    
    # If type is empty, default to texture
    if not type_val or type_val == '':
        type_val = 'texture'
    
    # Extract mood+index (third part)
    moodindex = parts[2] if len(parts) > 2 else 'unknown000'
    
    # Parse mood and index from compound
    # Pattern: moodname + 3 digits
    match = re.match(r'([a-z]+)(\d{3})', moodindex)
    if match:
        mood = match.group(1)
        index = match.group(2)
    else:
        mood = 'unknown'
        index = '000'
    
    return {
        'filename': filename.name,
        'category': category,
        'synth': synth,
        'type': type_val,
        'mood': mood,
        'index': index
    }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*60)
    print("LATENT SOUND ATLAS - FEATURE EXTRACTION (Librosa)")
    print("="*60)
    
    # Check audio files
    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    
    if len(audio_files) == 0:
        print(f"\n❌ ERROR: No audio files found in {AUDIO_DIR}")
        return
    
    print(f"\n✓ Found {len(audio_files)} audio files")
    
    # Test filename parsing
    print("\n=== TESTING FILENAME PARSING ===")
    for test_file in audio_files[:5]:
        parsed = parse_filename_metadata(test_file)
        print(f"{test_file.name} →")
        print(f"  Category: {parsed['category']}, Synth: {parsed['synth']}, "
              f"Type: {parsed['type']}, Mood: {parsed['mood']}, Index: {parsed['index']}")
    
    # Test feature extraction on first file
    print("\n=== TESTING FEATURE EXTRACTION ===")
    print(f"Testing on: {audio_files[0].name}")
    
    test_features = extract_features(audio_files[0])
    
    if test_features is None:
        print("❌ Feature extraction failed")
        return
    
    print(f"✓ Test successful! Extracted {len(test_features)} features")
    print("  Example features:", list(test_features.keys())[:5])
    
    # Extract features from all files
    print(f"\n=== EXTRACTING FEATURES FROM ALL FILES ===")
    all_features = []
    failed_files = []
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        # Get metadata
        metadata = parse_filename_metadata(audio_file)
        
        # Extract features
        features = extract_features(audio_file)
        
        if features is not None:
            # Combine metadata and features
            combined = {**metadata, **features}
            all_features.append(combined)
        else:
            failed_files.append(audio_file.name)
    
    if not all_features:
        print("\n❌ No features extracted successfully")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Save to CSV
    output_path = OUTPUT_DIR / "master_features.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("✓ FEATURE EXTRACTION COMPLETE!")
    print(f"{'='*60}")
    print(f"✓ Saved {len(df)} rows to {output_path}")
    print(f"✓ Total features per sound: {len(df.columns) - 6}")  # Minus metadata columns
    
    if failed_files:
        print(f"\n⚠ {len(failed_files)} files failed:")
        for f in failed_files[:5]:
            print(f"  - {f}")
    
    # Print summary
    print("\n=== FEATURE SUMMARY ===")
    print(f"Categories: {sorted(df['category'].unique())}")
    print(f"Synths: {sorted(df['synth'].unique())}")
    print(f"Moods: {sorted(df['mood'].unique())}")
    print(f"Types: {sorted(df['type'].unique())}")
    
    # Show feature statistics
    print("\n=== SAMPLE FEATURE STATISTICS ===")
    feature_cols = [col for col in df.columns if col not in ['filename', 'category', 'synth', 'type', 'mood', 'index']]
    print(f"Total features: {len(feature_cols)}")
    print("\nFirst few features:")
    for col in feature_cols[:5]:
        print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

if __name__ == "__main__":
    main()