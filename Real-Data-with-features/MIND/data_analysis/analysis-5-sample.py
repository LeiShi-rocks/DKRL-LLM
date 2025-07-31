#!/usr/bin/env python3
"""
SAMPLED MIND Click-Only User Feature Test
=========================================

Quick evaluation using only 1000 data points to test whether
click-based user features outperform impression-based features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("🎯 SAMPLED MIND CLICK-BASED USER FEATURE TEST")
print("=" * 60)
print("Using small sample for fast evaluation...")

# Load data
print("📂 Loading MIND data...")
behaviors_path = '../MINDsmall_train/behaviors.tsv'
news_path = '../MINDsmall_train/news.tsv'

behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None,
                          names=['impression_id', 'user_id', 'time', 'history', 'impressions'])

news_df = pd.read_csv(news_path, sep='\t', header=None,
                     names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])

print(f"✅ Full data loaded: {len(behaviors_df):,} behaviors, {len(news_df):,} news")

# SAMPLE THE DATA
print("🎲 Sampling data for fast evaluation...")
np.random.seed(42)

# Sample 1000 behavior records
sample_size = 1000
sample_indices = np.random.choice(len(behaviors_df), min(sample_size, len(behaviors_df)), replace=False)
behaviors_sample = behaviors_df.iloc[sample_indices].copy()

print(f"📊 Using sample: {len(behaviors_sample):,} behavior records")

# Parse timestamps
behaviors_sample['timestamp'] = pd.to_datetime(behaviors_sample['time'], format='%m/%d/%Y %I:%M:%S %p')
behaviors_sample['day_of_period'] = (behaviors_sample['timestamp'] - behaviors_sample['timestamp'].min()).dt.days + 1

# Split periods
train_mask = behaviors_sample['day_of_period'] <= 4
test_mask = behaviors_sample['day_of_period'] == 5

behaviors_train = behaviors_sample[train_mask].copy()
behaviors_test = behaviors_sample[test_mask].copy()

print(f"📅 Sample split - Training: {len(behaviors_train):,}, Test: {len(behaviors_test):,}")

# Simple embeddings based on categories
print("🎯 Creating simple news embeddings...")
categories = news_df['category'].fillna('unknown').unique()
subcategories = news_df['subcategory'].fillna('unknown').unique()

np.random.seed(42)
category_to_id = {cat: i for i, cat in enumerate(categories)}
subcat_to_id = {cat: i for i, cat in enumerate(subcategories)}

# Simple random embeddings
emb_dim = 50  # Smaller for sample
category_embeddings = np.random.normal(0, 0.1, (len(categories), emb_dim))
subcat_embeddings = np.random.normal(0, 0.1, (len(subcategories), emb_dim))

# Pre-compute news embeddings (only for news in our sample)
news_in_sample = set()
for _, row in behaviors_sample.iterrows():
    if pd.notna(row['history']):
        news_in_sample.update(row['history'].split())
    if pd.notna(row['impressions']):
        for item in row['impressions'].split():
            if '-' in item:
                news_id = item.split('-')[0]
                news_in_sample.add(news_id)

print(f"📊 Computing embeddings for {len(news_in_sample):,} news articles in sample...")

news_embeddings = {}
for news_id in news_in_sample:
    # Find news info
    news_row = news_df[news_df['news_id'] == news_id]
    if len(news_row) > 0:
        cat = news_row['category'].iloc[0]
        subcat = news_row['subcategory'].iloc[0]
        
        cat_id = category_to_id.get(cat, 0)
        subcat_id = subcat_to_id.get(subcat, 0)
        
        # Combine embeddings
        news_emb = (category_embeddings[cat_id] + subcat_embeddings[subcat_id]) / 2
        news_embeddings[news_id] = news_emb

print(f"✅ Created embeddings for {len(news_embeddings):,} news articles")

def extract_clicks_and_views(history_str, impressions_str):
    """Extract both clicks and all impressions (views)"""
    clicks = []
    views = []
    
    # History (all are clicks)
    if pd.notna(history_str) and history_str.strip():
        clicks.extend(history_str.split())
    
    # Impressions
    if pd.notna(impressions_str) and impressions_str.strip():
        for item in impressions_str.split():
            if '-' in item:
                news_id, click = item.rsplit('-', 1)
                views.append(news_id)
                if int(click) == 1:
                    clicks.append(news_id)
    
    return clicks, views

print("\n🔍 BUILDING USER PROFILES: CLICKS vs VIEWS")
print("=" * 50)

# Build both click-based and view-based user profiles
user_clicks_train = defaultdict(list)
user_views_train = defaultdict(list)

print("📊 Processing training data...")
for _, row in behaviors_train.iterrows():
    user_id = row['user_id']
    clicks, views = extract_clicks_and_views(row['history'], row['impressions'])
    
    # Collect click embeddings
    for news_id in clicks:
        if news_id in news_embeddings:
            user_clicks_train[user_id].append(news_embeddings[news_id])
    
    # Collect view embeddings (all impressions)
    for news_id in views:
        if news_id in news_embeddings:
            user_views_train[user_id].append(news_embeddings[news_id])

user_clicks_test = defaultdict(list)
user_views_test = defaultdict(list)

print("📊 Processing test data...")
for _, row in behaviors_test.iterrows():
    user_id = row['user_id']
    clicks, views = extract_clicks_and_views(row['history'], row['impressions'])
    
    for news_id in clicks:
        if news_id in news_embeddings:
            user_clicks_test[user_id].append(news_embeddings[news_id])
    
    for news_id in views:
        if news_id in news_embeddings:
            user_views_test[user_id].append(news_embeddings[news_id])

print(f"✅ Training users - Clicks: {len(user_clicks_train):,}, Views: {len(user_views_train):,}")
print(f"✅ Test users - Clicks: {len(user_clicks_test):,}, Views: {len(user_views_test):,}")

# Create user embeddings
def create_user_embedding(embeddings):
    """Create user embedding from news embeddings"""
    if not embeddings:
        return None
    
    embs = np.array(embeddings)
    # Simple aggregation: mean, std, count
    features = np.concatenate([
        np.mean(embs, axis=0),      # Average preferences
        np.std(embs, axis=0),       # Preference diversity
        [len(embs)]                 # Activity level
    ])
    return features

print("\n🧠 Creating user embeddings...")

# Click-based embeddings
train_user_click_embs = {}
for user_id, clicks in user_clicks_train.items():
    emb = create_user_embedding(clicks)
    if emb is not None:
        train_user_click_embs[user_id] = emb

test_user_click_embs = {}
for user_id, clicks in user_clicks_test.items():
    emb = create_user_embedding(clicks)
    if emb is not None:
        test_user_click_embs[user_id] = emb

# View-based embeddings (original approach)
train_user_view_embs = {}
for user_id, views in user_views_train.items():
    emb = create_user_embedding(views)
    if emb is not None:
        train_user_view_embs[user_id] = emb

test_user_view_embs = {}
for user_id, views in user_views_test.items():
    emb = create_user_embedding(views)
    if emb is not None:
        test_user_view_embs[user_id] = emb

if train_user_click_embs:
    user_emb_dim = len(next(iter(train_user_click_embs.values())))
    print(f"✅ User embedding dimension: {user_emb_dim}")
    print(f"   Click-based: {len(train_user_click_embs):,} train, {len(test_user_click_embs):,} test")
    print(f"   View-based: {len(train_user_view_embs):,} train, {len(test_user_view_embs):,} test")

# Create session-level data
print("\n🎯 Creating session-level data...")

def create_simple_sessions(behaviors_df, news_embeddings):
    """Create simple session data"""
    sessions = []
    
    for _, row in behaviors_df.iterrows():
        user_id = row['user_id']
        
        # Collect news embeddings and clicks for this record
        news_embs = []
        total_clicks = 0
        total_impressions = 0
        
        if pd.notna(row['impressions']):
            for item in row['impressions'].split():
                if '-' in item:
                    news_id, click = item.rsplit('-', 1)
                    if news_id in news_embeddings:
                        news_embs.append(news_embeddings[news_id])
                        total_impressions += 1
                        total_clicks += int(click)
        
        if news_embs:  # Only if we have valid news
            sessions.append({
                'user_id': user_id,
                'news_emb': np.mean(news_embs, axis=0),
                'ctr': total_clicks / max(total_impressions, 1)
            })
    
    return sessions

train_sessions = create_simple_sessions(behaviors_train, news_embeddings)
test_sessions = create_simple_sessions(behaviors_test, news_embeddings)

print(f"✅ Sessions: {len(train_sessions):,} train, {len(test_sessions):,} test")

if not train_sessions or not test_sessions:
    print("❌ No valid sessions created. Exiting.")
    exit(1)

# Convert to matrices
X_news_train = np.array([s['news_emb'] for s in train_sessions])
y_train = np.array([s['ctr'] for s in train_sessions])

X_news_test = np.array([s['news_emb'] for s in test_sessions])
y_test = np.array([s['ctr'] for s in test_sessions])

# Create user feature matrices
def get_user_features(sessions, user_embeddings, emb_dim):
    """Map sessions to user features"""
    user_features = []
    for session in sessions:
        user_id = session['user_id']
        if user_id in user_embeddings:
            user_features.append(user_embeddings[user_id])
        else:
            user_features.append(np.zeros(emb_dim))
    return np.array(user_features)

X_user_click_train = get_user_features(train_sessions, train_user_click_embs, user_emb_dim)
X_user_click_test = get_user_features(test_sessions, test_user_click_embs, user_emb_dim)

X_user_view_train = get_user_features(train_sessions, train_user_view_embs, user_emb_dim)
X_user_view_test = get_user_features(test_sessions, test_user_view_embs, user_emb_dim)

# Normalize
scaler_news = StandardScaler()
scaler_click = StandardScaler()
scaler_view = StandardScaler()

X_news_train_norm = scaler_news.fit_transform(X_news_train)
X_news_test_norm = scaler_news.transform(X_news_test)

X_user_click_train_norm = scaler_click.fit_transform(X_user_click_train)
X_user_click_test_norm = scaler_click.transform(X_user_click_test)

X_user_view_train_norm = scaler_view.fit_transform(X_user_view_train)
X_user_view_test_norm = scaler_view.transform(X_user_view_test)

print(f"📊 Feature matrices created:")
print(f"   News: {X_news_train_norm.shape}")
print(f"   User (Click): {X_user_click_train_norm.shape}")
print(f"   User (View): {X_user_view_train_norm.shape}")
print(f"   Target CTR range: [{y_train.min():.3f}, {y_train.max():.3f}]")

print("\n🏆 COMPARING CLICK-BASED vs VIEW-BASED USER FEATURES")
print("=" * 70)

results = {}

# Test different combinations
feature_combinations = [
    ('News_Only', X_news_train_norm, X_news_test_norm, 'News content only'),
    ('User_Click_Only', X_user_click_train_norm, X_user_click_test_norm, 'Click-based user features'),
    ('User_View_Only', X_user_view_train_norm, X_user_view_test_norm, 'View-based user features (original)'),
    ('News_Click_Combined', np.hstack([X_news_train_norm, X_user_click_train_norm]), 
     np.hstack([X_news_test_norm, X_user_click_test_norm]), 'News + Click-based user'),
    ('News_View_Combined', np.hstack([X_news_train_norm, X_user_view_train_norm]), 
     np.hstack([X_news_test_norm, X_user_view_test_norm]), 'News + View-based user (original)')
]

for name, X_train, X_test, description in feature_combinations:
    print(f"\n🧪 Testing: {description}")
    
    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1] if len(np.unique(y_pred)) > 1 else 0
    
    results[name] = {
        'MSE': mse,
        'R2': r2,
        'Correlation': corr,
        'Features': X_train.shape[1],
        'Description': description
    }
    
    print(f"   Ridge: R²={r2:.4f}, Corr={corr:.4f}, MSE={mse:.6f}")

print("\n📊 FINAL COMPARISON: CLICK vs VIEW USER FEATURES")
print("=" * 70)

df_results = pd.DataFrame(results).T
df_sorted = df_results.sort_values('R2', ascending=False)

print("🏆 PERFORMANCE RANKING:")
print("=" * 100)
print(df_sorted[['R2', 'Correlation', 'MSE', 'Features', 'Description']].round(4).to_string())

print(f"\n🔍 KEY INSIGHTS FROM SAMPLE:")
print("=" * 50)

click_r2 = results['User_Click_Only']['R2']
view_r2 = results['User_View_Only']['R2']
news_r2 = results['News_Only']['R2']

click_combined_r2 = results['News_Click_Combined']['R2']
view_combined_r2 = results['News_View_Combined']['R2']

print(f"📈 USER FEATURE COMPARISON:")
print(f"   Click-based user features: R² = {click_r2:.4f}")
print(f"   View-based user features:  R² = {view_r2:.4f}")
print(f"   Improvement: {click_r2 - view_r2:+.4f} ({((click_r2 - view_r2)/abs(view_r2)*100) if view_r2 != 0 else 0:+.1f}%)")

print(f"\n📈 COMBINED FEATURE COMPARISON:")
print(f"   News + Click-based: R² = {click_combined_r2:.4f}")
print(f"   News + View-based:  R² = {view_combined_r2:.4f}")
print(f"   Improvement: {click_combined_r2 - view_combined_r2:+.4f}")

print(f"\n💡 CONCLUSIONS:")
if click_r2 > view_r2:
    print(f"✅ HYPOTHESIS CONFIRMED: Click-based user features outperform view-based!")
    print(f"   This validates our finding that impression bias contaminates user modeling")
elif abs(click_r2 - view_r2) < 0.01:
    print(f"📊 SIMILAR PERFORMANCE: Click and view features perform similarly")
    print(f"   May indicate that the sample is too small or both approaches struggle")
else:
    print(f"❌ HYPOTHESIS NOT CONFIRMED: View-based features still better")
    print(f"   This could indicate issues with our click-based approach")

if click_combined_r2 > view_combined_r2:
    print(f"\n🎉 COMBINED FEATURES: Click-based approach wins overall!")
else:
    print(f"\n📊 COMBINED FEATURES: View-based approach still competitive")

best_model = df_sorted.iloc[0]
print(f"\n🥇 BEST OVERALL: {best_model.name}")
print(f"   R² = {best_model['R2']:.4f}")
print(f"   Strategy: {best_model['Description']}")

# Quick visualization
plt.figure(figsize=(12, 4))

# Performance comparison
plt.subplot(1, 2, 1)
models = ['News\nOnly', 'User\n(Click)', 'User\n(View)', 'News+Click', 'News+View']
r2_values = [results['News_Only']['R2'], results['User_Click_Only']['R2'], 
             results['User_View_Only']['R2'], results['News_Click_Combined']['R2'],
             results['News_View_Combined']['R2']]

colors = ['blue', 'green', 'red', 'darkgreen', 'darkred']
bars = plt.bar(models, r2_values, color=colors, alpha=0.7)

plt.ylabel('R² Score')
plt.title('Feature Type Comparison (Sample)')
plt.grid(True, alpha=0.3)

for bar, r2 in zip(bars, r2_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{r2:.3f}', ha='center', va='bottom', fontsize=9)

# User feature direct comparison
plt.subplot(1, 2, 2)
user_types = ['Click-based\nUser Features', 'View-based\nUser Features']
user_r2 = [click_r2, view_r2]
user_colors = ['green' if click_r2 > view_r2 else 'orange', 'red' if click_r2 > view_r2 else 'blue']

bars2 = plt.bar(user_types, user_r2, color=user_colors, alpha=0.7)
plt.ylabel('R² Score')
plt.title('Click vs View User Features')
plt.grid(True, alpha=0.3)

for bar, r2 in zip(bars2, user_r2):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{r2:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print(f"\n🎊 SAMPLED ANALYSIS COMPLETE!")
print(f"   Sample size: {len(behaviors_sample):,} records")
print(f"   Key finding: {'Click-based approach superior' if click_r2 > view_r2 else 'Results inconclusive - may need larger sample'}")

# Save results
df_results.to_csv('sampled_click_vs_view_results.csv')
print(f"💾 Results saved to sampled_click_vs_view_results.csv") 