#!/usr/bin/env python3
"""
Filter Unique News Embeddings and User Features
=============================================

This script processes X_news_embeddings.npy and X_user_features.npy to:
1. Load and examine the data structure
2. Filter out unique news embeddings 
3. Filter out unique user features
4. Save the filtered results
"""

import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import time
from collections import defaultdict

print("🔍 PROCESSING MIND EMBEDDINGS FOR UNIQUE FILTERING")
print("=" * 60)

def analyze_data_structure(data, name):
    """Analyze the structure of the data"""
    print(f"\n📊 {name} Analysis:")
    print("-" * 30)
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Memory usage: {data.nbytes / (1024**2):.1f} MB")
    print(f"Min value: {data.min():.6f}")
    print(f"Max value: {data.max():.6f}")
    print(f"Mean: {data.mean():.6f}")
    print(f"Std: {data.std():.6f}")
    
    # Check for duplicates
    unique_rows = np.unique(data, axis=0)
    print(f"Original rows: {data.shape[0]:,}")
    print(f"Unique rows: {unique_rows.shape[0]:,}")
    print(f"Duplicate rows: {data.shape[0] - unique_rows.shape[0]:,}")
    print(f"Uniqueness ratio: {unique_rows.shape[0] / data.shape[0]:.4f}")
    
    return unique_rows

def find_unique_with_mapping(data, name):
    """Find unique rows and create mapping from original to unique indices"""
    print(f"\n🔍 Finding unique {name} with index mapping...")
    start_time = time.time()
    
    # Find unique rows and return indices
    unique_data, unique_indices, inverse_indices = np.unique(
        data, axis=0, return_index=True, return_inverse=True
    )
    
    processing_time = time.time() - start_time
    
    print(f"✅ Processing completed in {processing_time:.2f} seconds")
    print(f"Original data: {data.shape}")
    print(f"Unique data: {unique_data.shape}")
    print(f"Compression ratio: {unique_data.shape[0] / data.shape[0]:.4f}")
    
    # Create mapping dictionaries
    original_to_unique = {}  # Maps original index to unique index
    unique_to_original = defaultdict(list)  # Maps unique index to list of original indices
    
    for original_idx, unique_idx in enumerate(inverse_indices):
        original_to_unique[original_idx] = unique_idx
        unique_to_original[unique_idx].append(original_idx)
    
    mapping_info = {
        'original_shape': [int(x) for x in data.shape],
        'unique_shape': [int(x) for x in unique_data.shape],
        'compression_ratio': float(unique_data.shape[0] / data.shape[0]),
        'original_to_unique': {str(k): int(v) for k, v in original_to_unique.items()},
        'unique_to_original': {str(k): [int(x) for x in v] for k, v in unique_to_original.items()},
        'inverse_indices': [int(x) for x in inverse_indices],
        'unique_indices': [int(x) for x in unique_indices]
    }
    
    return unique_data, mapping_info

# Load the data files
print("📂 Loading data files...")

try:
    X_news = np.load('X_news_embeddings.npy')
    print(f"✅ Loaded X_news_embeddings.npy: {X_news.shape}")
except Exception as e:
    print(f"❌ Error loading X_news_embeddings.npy: {e}")
    X_news = None

try:
    X_user = np.load('X_user_features.npy')
    print(f"✅ Loaded X_user_features.npy: {X_user.shape}")
except Exception as e:
    print(f"❌ Error loading X_user_features.npy: {e}")
    X_user = None

try:
    y_clicks = np.load('y_clicks.npy')
    print(f"✅ Loaded y_clicks.npy: {y_clicks.shape}")
except Exception as e:
    print(f"❌ Error loading y_clicks.npy: {e}")
    y_clicks = None

# Process news embeddings
if X_news is not None:
    print("\n" + "="*60)
    print("🗞️  PROCESSING NEWS EMBEDDINGS")
    print("="*60)
    
    # Analyze structure
    analyze_data_structure(X_news, "News Embeddings")
    
    # Find unique news embeddings
    X_news_unique, news_mapping = find_unique_with_mapping(X_news, "news embeddings")
    
    # Save unique news embeddings
    np.save('X_news_embeddings_unique.npy', X_news_unique)
    print(f"💾 Saved unique news embeddings to X_news_embeddings_unique.npy")
    
    # Save mapping information
    with open('news_embeddings_mapping.json', 'w') as f:
        json.dump(news_mapping, f, indent=2)
    print(f"💾 Saved news mapping to news_embeddings_mapping.json")
    
else:
    print("⚠️  Skipping news embeddings processing due to loading error")

# Process user features
if X_user is not None:
    print("\n" + "="*60)
    print("👤 PROCESSING USER FEATURES") 
    print("="*60)
    
    # Analyze structure
    analyze_data_structure(X_user, "User Features")
    
    # Find unique user features
    X_user_unique, user_mapping = find_unique_with_mapping(X_user, "user features")
    
    # Save unique user features
    np.save('X_user_features_unique.npy', X_user_unique)
    print(f"💾 Saved unique user features to X_user_features_unique.npy")
    
    # Save mapping information
    with open('user_features_mapping.json', 'w') as f:
        json.dump(user_mapping, f, indent=2)
    print(f"💾 Saved user mapping to user_features_mapping.json")
    
else:
    print("⚠️  Skipping user features processing due to loading error")

# Create summary report
print("\n" + "="*60)
print("📋 PROCESSING SUMMARY")
print("="*60)

summary = {
    'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'original_files': {
        'X_news_embeddings.npy': {
            'shape': [int(x) for x in X_news.shape] if X_news is not None else None,
            'size_mb': float(X_news.nbytes / (1024**2)) if X_news is not None else None
        },
        'X_user_features.npy': {
            'shape': [int(x) for x in X_user.shape] if X_user is not None else None,
            'size_mb': float(X_user.nbytes / (1024**2)) if X_user is not None else None
        }
    },
    'filtered_files': {}
}

if X_news is not None:
    summary['filtered_files']['X_news_embeddings_unique.npy'] = {
        'original_shape': [int(x) for x in X_news.shape],
        'unique_shape': [int(x) for x in X_news_unique.shape],
        'compression_ratio': float(X_news_unique.shape[0] / X_news.shape[0]),
        'size_reduction_mb': float((X_news.nbytes - X_news_unique.nbytes) / (1024**2)),
        'duplicates_removed': int(X_news.shape[0] - X_news_unique.shape[0])
    }
    
    print(f"🗞️  News Embeddings:")
    print(f"   Original: {X_news.shape[0]:,} embeddings ({X_news.nbytes/(1024**2):.1f} MB)")
    print(f"   Unique: {X_news_unique.shape[0]:,} embeddings ({X_news_unique.nbytes/(1024**2):.1f} MB)")
    print(f"   Removed: {X_news.shape[0] - X_news_unique.shape[0]:,} duplicates")
    print(f"   Compression: {X_news_unique.shape[0] / X_news.shape[0]:.4f}")
    print(f"   Space saved: {(X_news.nbytes - X_news_unique.nbytes)/(1024**2):.1f} MB")

if X_user is not None:
    summary['filtered_files']['X_user_features_unique.npy'] = {
        'original_shape': [int(x) for x in X_user.shape],
        'unique_shape': [int(x) for x in X_user_unique.shape],
        'compression_ratio': float(X_user_unique.shape[0] / X_user.shape[0]),
        'size_reduction_mb': float((X_user.nbytes - X_user_unique.nbytes) / (1024**2)),
        'duplicates_removed': int(X_user.shape[0] - X_user_unique.shape[0])
    }
    
    print(f"👤 User Features:")
    print(f"   Original: {X_user.shape[0]:,} features ({X_user.nbytes/(1024**2):.1f} MB)")
    print(f"   Unique: {X_user_unique.shape[0]:,} features ({X_user_unique.nbytes/(1024**2):.1f} MB)")
    print(f"   Removed: {X_user.shape[0] - X_user_unique.shape[0]:,} duplicates")
    print(f"   Compression: {X_user_unique.shape[0] / X_user.shape[0]:.4f}")
    print(f"   Space saved: {(X_user.nbytes - X_user_unique.nbytes)/(1024**2):.1f} MB")

# Save summary
with open('filtering_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n💾 Complete summary saved to filtering_summary.json")
print(f"\n🎉 FILTERING COMPLETE!")
print("\nGenerated files:")
print("  📁 X_news_embeddings_unique.npy")
print("  📁 X_user_features_unique.npy") 
print("  📁 news_embeddings_mapping.json")
print("  📁 user_features_mapping.json")
print("  📁 filtering_summary.json") 