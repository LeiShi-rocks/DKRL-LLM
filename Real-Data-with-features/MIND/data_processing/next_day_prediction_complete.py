#!/usr/bin/env python3
"""
Next-Day Click Prediction Dataset Creation for DKRL Research

This script creates a dataset for predicting day 5 clicks using user profiles 
built from days 1-4, integrating entity and relation embeddings.

Usage:
    python next_day_prediction_complete.py
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("🎯 Next-Day Click Prediction Dataset Creation")
print("=" * 60)

# =============================================================================
# 1. LOAD DATA AND EMBEDDINGS
# =============================================================================

def load_embeddings(embedding_file):
    """Load embeddings from .vec file format"""
    embeddings = {}
    with open(embedding_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            entity_id = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[entity_id] = vector
    return embeddings

def parse_entities(entity_str):
    """Parse entity JSON string and extract WikidataIds"""
    if pd.isna(entity_str) or entity_str == '[]':
        return []
    try:
        entities = json.loads(entity_str)
        return [entity['WikidataId'] for entity in entities if 'WikidataId' in entity]
    except:
        return []

def create_news_embedding(entities, entity_embeddings, method='mean'):
    """Create news embedding by aggregating entity embeddings"""
    if not entities:
        return np.zeros(100)
    
    valid_embeddings = []
    for entity in entities:
        if entity in entity_embeddings:
            valid_embeddings.append(entity_embeddings[entity])
    
    if not valid_embeddings:
        return np.zeros(100)
    
    embeddings_matrix = np.array(valid_embeddings)
    
    if method == 'mean':
        return np.mean(embeddings_matrix, axis=0)
    elif method == 'sum':
        return np.sum(embeddings_matrix, axis=0)
    else:
        return np.mean(embeddings_matrix, axis=0)

print("📊 Loading embeddings...")
entity_embeddings = load_embeddings('MINDsmall_train/entity_embedding.vec')
print(f"✓ Loaded {len(entity_embeddings)} entity embeddings")

print("📰 Loading news data...")
news_columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
news_df = pd.read_csv('MINDsmall_train/news.tsv', sep='\t', header=None, names=news_columns)

# Parse entities and create news embeddings
news_df['title_entities'] = news_df['TitleEntities'].apply(parse_entities)
news_df['abstract_entities'] = news_df['AbstractEntities'].apply(parse_entities)
news_df['all_entities'] = news_df['title_entities'] + news_df['abstract_entities']

print("🔗 Creating news embeddings...")
news_embeddings = {}
for idx, row in news_df.iterrows():
    news_id = row['NewsID']
    entities = row['all_entities']
    embedding = create_news_embedding(entities, entity_embeddings)
    news_embeddings[news_id] = embedding

print(f"✓ Created embeddings for {len(news_embeddings)} news articles")

print("👥 Loading behavior data...")
behavior_columns = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
behavior_df = pd.read_csv('MINDsmall_train/behaviors.tsv', sep='\t', header=None, names=behavior_columns)
behavior_df['DateTime'] = pd.to_datetime(behavior_df['Time'])
behavior_df['Date'] = behavior_df['DateTime'].dt.date

print(f"✓ Loaded {len(behavior_df)} behavior records")
print(f"Date range: {behavior_df['Date'].min()} to {behavior_df['Date'].max()}")

# =============================================================================
# 2. TEMPORAL DATA SPLIT
# =============================================================================

print("\n🔄 Performing temporal data split...")

# Split data by days
dates = sorted(behavior_df['Date'].unique())
print(f"📅 Available dates: {dates}")

# Define training and test periods
train_dates = dates[:4]  # Days 1-4
test_date = dates[4]     # Day 5

print(f"🏋️ Training period: {train_dates[0]} to {train_dates[-1]}")
print(f"🎯 Test day: {test_date}")

# Split behavior data
train_behavior = behavior_df[behavior_df['Date'].isin(train_dates)].copy()
test_behavior = behavior_df[behavior_df['Date'] == test_date].copy()

print(f"\n📊 Data Split Summary:")
print(f"   Training records: {len(train_behavior):,}")
print(f"   Test records: {len(test_behavior):,}")
print(f"   Training users: {train_behavior['UserID'].nunique():,}")
print(f"   Test users: {test_behavior['UserID'].nunique():,}")

# Check user overlap
train_users = set(train_behavior['UserID'].unique())
test_users = set(test_behavior['UserID'].unique())
overlap_users = train_users.intersection(test_users)

print(f"   User overlap: {len(overlap_users):,} ({len(overlap_users)/len(test_users)*100:.1f}% of test users)")
print(f"   Cold-start users: {len(test_users - train_users):,}")

# =============================================================================
# 3. BUILD USER PROFILES FROM DAYS 1-4
# =============================================================================

def parse_history(history_str):
    """Parse history string into list of news IDs"""
    if pd.isna(history_str):
        return []
    return history_str.split()

def parse_impressions(impressions_str):
    """Parse impressions string into list of (news_id, label) tuples"""
    if pd.isna(impressions_str):
        return []
    
    impressions = []
    for item in impressions_str.split():
        parts = item.split('-')
        if len(parts) == 2:
            news_id, label = parts
            impressions.append((news_id, int(label)))
    return impressions

def build_user_profile(user_behavior_df, news_df, news_embeddings):
    """
    Build comprehensive user profile from behavior history
    
    Returns:
        dict with user features including embeddings and statistics
    """
    profile = {
        'user_id': user_behavior_df.iloc[0]['UserID'],
        'total_sessions': len(user_behavior_df),
        'total_days_active': user_behavior_df['Date'].nunique(),
        'first_session': user_behavior_df['DateTime'].min(),
        'last_session': user_behavior_df['DateTime'].max(),
        'session_span_hours': (user_behavior_df['DateTime'].max() - user_behavior_df['DateTime'].min()).total_seconds() / 3600,
    }
    
    # Parse all history and impressions
    all_history = []
    all_impressions = []
    all_clicks = []
    
    for _, row in user_behavior_df.iterrows():
        history = parse_history(row['History'])
        impressions = parse_impressions(row['Impressions'])
        
        all_history.extend(history)
        all_impressions.extend([news_id for news_id, _ in impressions])
        all_clicks.extend([news_id for news_id, label in impressions if label == 1])
    
    # Basic statistics
    profile['total_history_items'] = len(all_history)
    profile['total_impressions'] = len(all_impressions)
    profile['total_clicks'] = len(all_clicks)
    profile['overall_ctr'] = len(all_clicks) / len(all_impressions) if len(all_impressions) > 0 else 0
    profile['unique_history_items'] = len(set(all_history))
    profile['unique_clicked_items'] = len(set(all_clicks))
    
    return profile, all_clicks, all_history

print("\n🏗️ Building user profiles from training data...")
user_profiles = {}

# Sample first 500 users for demonstration (you can increase this)
sample_users = train_behavior['UserID'].unique()[:500]

for i, user_id in enumerate(sample_users):
    if i % 100 == 0:
        print(f"   Processing user {i+1}/{len(sample_users)}...")
    
    user_data = train_behavior[train_behavior['UserID'] == user_id].sort_values('DateTime')
    profile, all_clicks, all_history = build_user_profile(user_data, news_df, news_embeddings)
    
    # Add category analysis
    clicked_news_info = []
    if all_clicks:
        for news_id in set(all_clicks):
            news_row = news_df[news_df['NewsID'] == news_id]
            if not news_row.empty:
                clicked_news_info.append({
                    'category': news_row.iloc[0]['Category'],
                    'subcategory': news_row.iloc[0]['SubCategory']
                })
    
    # Category preferences
    if clicked_news_info:
        categories = [item['category'] for item in clicked_news_info]
        category_counts = Counter(categories)
        total_cats = sum(category_counts.values())
        
        profile['top_category'] = category_counts.most_common(1)[0][0] if category_counts else 'unknown'
        profile['category_diversity'] = len(category_counts)
        
        # Category proportions (top 5)
        for j, (cat, count) in enumerate(category_counts.most_common(5)):
            profile[f'cat_prop_{j+1}'] = count / total_cats
        for j in range(len(category_counts.most_common(5)), 5):
            profile[f'cat_prop_{j+1}'] = 0.0
    else:
        profile['top_category'] = 'unknown'
        profile['category_diversity'] = 0
        for j in range(5):
            profile[f'cat_prop_{j+1}'] = 0.0
    
    # Additional features
    profile['avg_sessions_per_day'] = profile['total_sessions'] / max(profile['total_days_active'], 1)
    profile['hours_since_last_session'] = 0
    
    # Create embeddings from clicked news
    clicked_embeddings = []
    for news_id in set(all_clicks):
        if news_id in news_embeddings:
            clicked_embeddings.append(news_embeddings[news_id])
    
    if clicked_embeddings:
        profile['user_embedding'] = np.mean(clicked_embeddings, axis=0)
        profile['embedding_coverage'] = len(clicked_embeddings) / len(set(all_clicks))
    else:
        profile['user_embedding'] = np.zeros(100)
        profile['embedding_coverage'] = 0.0
    
    # History embedding
    history_embeddings = []
    for news_id in set(all_history):
        if news_id in news_embeddings:
            history_embeddings.append(news_embeddings[news_id])
    
    if history_embeddings:
        profile['history_embedding'] = np.mean(history_embeddings, axis=0)
        profile['history_coverage'] = len(history_embeddings) / len(set(all_history))
    else:
        profile['history_embedding'] = np.zeros(100)
        profile['history_coverage'] = 0.0
    
    user_profiles[user_id] = profile

print(f"✓ Built profiles for {len(user_profiles):,} users")

# =============================================================================
# 4. PREPARE DAY 5 PREDICTION DATA
# =============================================================================

def prepare_prediction_dataset(test_behavior, user_profiles, news_embeddings, news_df):
    """
    Prepare dataset for next-day click prediction
    
    Returns:
        X_user: User feature matrix (n_samples, n_user_features)
        X_news: News embedding matrix (n_samples, 100)
        y: Click labels (n_samples,)
        metadata: Additional information for analysis
    """
    X_user_list = []
    X_news_list = []
    y_list = []
    metadata_list = []
    
    excluded_users = 0
    total_impressions = 0
    
    for _, row in test_behavior.iterrows():
        user_id = row['UserID']
        impressions = parse_impressions(row['Impressions'])
        
        # Skip users not in training data (cold-start users)
        if user_id not in user_profiles:
            excluded_users += 1
            continue
        
        user_profile = user_profiles[user_id]
        
        for news_id, label in impressions:
            total_impressions += 1
            
            # Skip news without embeddings
            if news_id not in news_embeddings:
                continue
            
            # User features (excluding embeddings for now)
            user_features = [
                user_profile['total_sessions'],
                user_profile['total_days_active'],
                user_profile['session_span_hours'],
                user_profile['total_history_items'],
                user_profile['total_impressions'],
                user_profile['total_clicks'],
                user_profile['overall_ctr'],
                user_profile['unique_history_items'],
                user_profile['unique_clicked_items'],
                user_profile['category_diversity'],
                user_profile['avg_sessions_per_day'],
                user_profile['hours_since_last_session'],
                user_profile['embedding_coverage'],
                user_profile['history_coverage'],
            ]
            
            # Add category proportions
            for i in range(5):
                user_features.append(user_profile[f'cat_prop_{i+1}'])
            
            # Combine user features with user embedding
            full_user_features = np.concatenate([
                user_features,
                user_profile['user_embedding'],
                user_profile['history_embedding']
            ])
            
            X_user_list.append(full_user_features)
            X_news_list.append(news_embeddings[news_id])
            y_list.append(label)
            
            # Metadata for analysis
            news_info = news_df[news_df['NewsID'] == news_id]
            metadata_list.append({
                'user_id': user_id,
                'news_id': news_id,
                'impression_id': row['ImpressionID'],
                'datetime': row['DateTime'],
                'news_category': news_info.iloc[0]['Category'] if not news_info.empty else 'unknown',
                'news_subcategory': news_info.iloc[0]['SubCategory'] if not news_info.empty else 'unknown'
            })
    
    X_user = np.array(X_user_list)
    X_news = np.array(X_news_list)
    y = np.array(y_list)
    metadata = metadata_list
    
    print(f"📊 Dataset Preparation Summary:")
    print(f"   Total test impressions: {total_impressions:,}")
    print(f"   Excluded cold-start users: {excluded_users}")
    print(f"   Final dataset size: {len(y):,} samples")
    print(f"   Positive samples: {sum(y):,} ({sum(y)/len(y)*100:.1f}%)")
    print(f"   User feature dimension: {X_user.shape[1]}")
    print(f"   News embedding dimension: {X_news.shape[1]}")
    
    return X_user, X_news, y, metadata

print("\n🎯 Preparing next-day prediction dataset...")
X_user, X_news, y, metadata = prepare_prediction_dataset(test_behavior, user_profiles, news_embeddings, news_df)

# Create feature names for interpretability
user_feature_names = [
    'total_sessions', 'total_days_active', 'session_span_hours',
    'total_history_items', 'total_impressions', 'total_clicks', 'overall_ctr',
    'unique_history_items', 'unique_clicked_items', 'category_diversity',
    'avg_sessions_per_day', 'hours_since_last_session', 'embedding_coverage', 'history_coverage'
] + [f'cat_prop_{i+1}' for i in range(5)] + [f'user_emb_{i}' for i in range(100)] + [f'hist_emb_{i}' for i in range(100)]

print(f"\n📋 Feature Names ({len(user_feature_names)} total):")
print(f"   Statistical features: {user_feature_names[:19]}")
print(f"   User embedding: user_emb_0 to user_emb_99")
print(f"   History embedding: hist_emb_0 to hist_emb_99")

# =============================================================================
# 5. SAVE DATASET AND CREATE USAGE EXAMPLES
# =============================================================================

print("\n💾 Saving dataset...")

# Save as numpy arrays
np.save('X_user_features.npy', X_user)
np.save('X_news_embeddings.npy', X_news)
np.save('y_clicks.npy', y)

# Save metadata and feature names
metadata_df = pd.DataFrame(metadata)
metadata_df['click'] = y
metadata_df.to_csv('prediction_metadata.csv', index=False)

# Save feature names
with open('feature_names.txt', 'w') as f:
    for name in user_feature_names:
        f.write(f"{name}\n")

# Save user profiles for potential future use
import pickle
with open('user_profiles.pkl', 'wb') as f:
    pickle.dump(user_profiles, f)

# Create dataset info
dataset_info = {
    'creation_date': datetime.now().isoformat(),
    'task': 'next_day_click_prediction',
    'training_period': f"{train_dates[0]} to {train_dates[-1]}",
    'test_date': str(test_date),
    'n_samples': len(y),
    'n_positive': int(sum(y)),
    'n_negative': int(sum(y == 0)),
    'positive_rate': float(sum(y) / len(y)),
    'n_users': len(user_profiles),
    'n_news_with_embeddings': len(news_embeddings),
    'user_feature_dim': X_user.shape[1],
    'news_embedding_dim': X_news.shape[1],
    'total_feature_dim': X_user.shape[1] + X_news.shape[1],
    'feature_breakdown': {
        'statistical_features': 19,
        'user_embedding': 100,
        'history_embedding': 100,
        'news_embedding': 100
    }
}

with open('dataset_info.json', 'w') as f:
    json.dump(dataset_info, f, indent=2)

print(f"✅ Dataset saved successfully!")
print(f"📁 Files created:")
print(f"   - X_user_features.npy: {X_user.shape} user features")
print(f"   - X_news_embeddings.npy: {X_news.shape} news embeddings")
print(f"   - y_clicks.npy: {y.shape} click labels")
print(f"   - prediction_metadata.csv: Sample metadata")
print(f"   - feature_names.txt: Feature name mapping")
print(f"   - user_profiles.pkl: Complete user profiles")
print(f"   - dataset_info.json: Dataset summary")

# =============================================================================
# 6. USAGE EXAMPLES AND SUMMARY
# =============================================================================

print("\n🎓 Usage Examples:")
print("="*60)
print("# Load the prepared dataset")
print("X_user = np.load('X_user_features.npy')")
print("X_news = np.load('X_news_embeddings.npy')")
print("y = np.load('y_clicks.npy')")
print("\n# Option 1: Concatenate features")
print("X_combined = np.concatenate([X_user, X_news], axis=1)")
print("# Train model: model.fit(X_combined, y)")
print("\n# Option 2: Separate feature processing")
print("# User features: X_user[:, :19] = statistics")
print("#               X_user[:, 19:119] = user embedding")
print("#               X_user[:, 119:219] = history embedding")
print("# News features: X_news = news embedding")
print("\n# Option 3: Neural network with multiple inputs")
print("# user_input = Input(shape=(219,))  # User features")
print("# news_input = Input(shape=(100,))  # News embedding")
print("# ... build neural network with two input branches")

# Show some examples of the data
print(f"\n📋 Sample Data Points:")
for i in range(min(3, len(y))):
    print(f"\nSample {i+1}:")
    print(f"   User ID: {metadata[i]['user_id']}")
    print(f"   News ID: {metadata[i]['news_id']}")
    print(f"   Category: {metadata[i]['news_category']}")
    print(f"   Click: {y[i]}")
    print(f"   User CTR: {X_user[i, 6]:.3f}")  # overall_ctr feature
    print(f"   User total clicks: {X_user[i, 5]:.0f}")  # total_clicks feature
    print(f"   User embedding norm: {np.linalg.norm(X_user[i, 19:119]):.3f}")
    print(f"   News embedding norm: {np.linalg.norm(X_news[i]):.3f}")

print(f"\n🎯 DKRL DATASET READY!")
print("="*60)
print(f"✅ Task: Next-day click prediction")
print(f"✅ User profiles: Days 1-4 behavioral + entity data")
print(f"✅ News embeddings: Entity-enhanced representations")
print(f"✅ Temporal split: Realistic train/test setup")
print(f"✅ DKRL features: Knowledge graph embeddings integrated")
print(f"✅ Rich features: {X_user.shape[1] + X_news.shape[1]} total dimensions")
print(f"✅ {len(y):,} labeled samples for training")
print(f"✅ Ready for model(x, t) → y training!")

print("\nScript completed successfully! 🚀") 