#!/usr/bin/env python3
"""
MIND Recency Test: Day 4 → Day 5
================================

Test whether using only day 4 to predict day 5 (instead of days 1-4 → day 5)
improves user feature performance by reducing temporal distance.
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

print("⏰ MIND RECENCY TEST: DAY 4 → DAY 5")
print("=" * 50)
print("Testing whether shorter temporal gap improves user features...")

# Load data
print("📂 Loading MIND data...")
behaviors_path = '../MINDsmall_train/behaviors.tsv'
news_path = '../MINDsmall_train/news.tsv'

behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None,
                          names=['impression_id', 'user_id', 'time', 'history', 'impressions'])

news_df = pd.read_csv(news_path, sep='\t', header=None,
                     names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])

print(f"✅ Data loaded: {len(behaviors_df):,} behaviors, {len(news_df):,} news")

# Parse timestamps
behaviors_df['timestamp'] = pd.to_datetime(behaviors_df['time'], format='%m/%d/%Y %I:%M:%S %p')
behaviors_df['day_of_period'] = (behaviors_df['timestamp'] - behaviors_df['timestamp'].min()).dt.days + 1

# NEW SPLIT: Day 4 → Day 5 (instead of Days 1-4 → Day 5)
day4_mask = behaviors_df['day_of_period'] == 4
day5_mask = behaviors_df['day_of_period'] == 5

behaviors_day4 = behaviors_df[day4_mask].copy()
behaviors_day5 = behaviors_df[day5_mask].copy()

print(f"⏰ NEW TEMPORAL SPLIT:")
print(f"   Training (Day 4 only): {len(behaviors_day4):,} records")
print(f"   Test (Day 5): {len(behaviors_day5):,} records")
print(f"   Temporal gap: 1 day (vs previous 4-day gap)")

# Sample for fast evaluation
print("\n🎲 Sampling for fast evaluation...")
np.random.seed(42)
sample_size = 1000

# Sample from both days
sample_day4_indices = np.random.choice(len(behaviors_day4), min(sample_size//2, len(behaviors_day4)), replace=False)
sample_day5_indices = np.random.choice(len(behaviors_day5), min(sample_size//2, len(behaviors_day5)), replace=False)

behaviors_train = behaviors_day4.iloc[sample_day4_indices].copy()
behaviors_test = behaviors_day5.iloc[sample_day5_indices].copy()

print(f"📊 Sample sizes:")
print(f"   Training (Day 4): {len(behaviors_train):,}")
print(f"   Test (Day 5): {len(behaviors_test):,}")

# Simple embeddings
print("\n🎯 Creating simple news embeddings...")
categories = news_df['category'].fillna('unknown').unique()
subcategories = news_df['subcategory'].fillna('unknown').unique()

np.random.seed(42)
category_to_id = {cat: i for i, cat in enumerate(categories)}
subcat_to_id = {cat: i for i, cat in enumerate(subcategories)}

emb_dim = 50
category_embeddings = np.random.normal(0, 0.1, (len(categories), emb_dim))
subcat_embeddings = np.random.normal(0, 0.1, (len(subcategories), emb_dim))

# Get news in sample
news_in_sample = set()
for _, row in pd.concat([behaviors_train, behaviors_test]).iterrows():
    if pd.notna(row['history']):
        news_in_sample.update(row['history'].split())
    if pd.notna(row['impressions']):
        for item in row['impressions'].split():
            if '-' in item:
                news_id = item.split('-')[0]
                news_in_sample.add(news_id)

print(f"📊 Computing embeddings for {len(news_in_sample):,} news articles...")

news_embeddings = {}
for news_id in news_in_sample:
    news_row = news_df[news_df['news_id'] == news_id]
    if len(news_row) > 0:
        cat = news_row['category'].iloc[0]
        subcat = news_row['subcategory'].iloc[0]
        
        cat_id = category_to_id.get(cat, 0)
        subcat_id = subcat_to_id.get(subcat, 0)
        
        news_emb = (category_embeddings[cat_id] + subcat_embeddings[subcat_id]) / 2
        news_embeddings[news_id] = news_emb

print(f"✅ Created embeddings for {len(news_embeddings):,} news articles")

def extract_clicks_and_views(history_str, impressions_str):
    """Extract both clicks and views"""
    clicks = []
    views = []
    
    if pd.notna(history_str) and history_str.strip():
        clicks.extend(history_str.split())
    
    if pd.notna(impressions_str) and impressions_str.strip():
        for item in impressions_str.split():
            if '-' in item:
                news_id, click = item.rsplit('-', 1)
                views.append(news_id)
                if int(click) == 1:
                    clicks.append(news_id)
    
    return clicks, views

print("\n🔍 BUILDING USER PROFILES: DAY 4 BEHAVIOR")
print("=" * 50)

# Build user profiles from Day 4 ONLY
user_clicks_day4 = defaultdict(list)
user_views_day4 = defaultdict(list)

print("📊 Processing Day 4 user behavior...")
for _, row in behaviors_train.iterrows():
    user_id = row['user_id']
    clicks, views = extract_clicks_and_views(row['history'], row['impressions'])
    
    for news_id in clicks:
        if news_id in news_embeddings:
            user_clicks_day4[user_id].append(news_embeddings[news_id])
    
    for news_id in views:
        if news_id in news_embeddings:
            user_views_day4[user_id].append(news_embeddings[news_id])

print(f"✅ Day 4 user profiles:")
print(f"   Users with clicks: {len(user_clicks_day4):,}")
print(f"   Users with views: {len(user_views_day4):,}")

def create_user_embedding(embeddings):
    """Create user embedding from news embeddings"""
    if not embeddings:
        return None
    
    embs = np.array(embeddings)
    features = np.concatenate([
        np.mean(embs, axis=0),      # Average preferences
        np.std(embs, axis=0),       # Preference diversity
        [len(embs)]                 # Activity level
    ])
    return features

# Create user embeddings from Day 4
print("\n🧠 Creating user embeddings from Day 4 behavior...")

day4_user_click_embs = {}
for user_id, clicks in user_clicks_day4.items():
    emb = create_user_embedding(clicks)
    if emb is not None:
        day4_user_click_embs[user_id] = emb

day4_user_view_embs = {}
for user_id, views in user_views_day4.items():
    emb = create_user_embedding(views)
    if emb is not None:
        day4_user_view_embs[user_id] = emb

if day4_user_click_embs:
    user_emb_dim = len(next(iter(day4_user_click_embs.values())))
    print(f"✅ User embedding dimension: {user_emb_dim}")
    print(f"   Click-based (Day 4): {len(day4_user_click_embs):,} users")
    print(f"   View-based (Day 4): {len(day4_user_view_embs):,} users")

# Create Day 5 sessions for prediction
print("\n🎯 Creating Day 5 sessions for prediction...")

def create_sessions(behaviors_df, news_embeddings):
    """Create session data"""
    sessions = []
    
    for _, row in behaviors_df.iterrows():
        user_id = row['user_id']
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
        
        if news_embs:
            sessions.append({
                'user_id': user_id,
                'news_emb': np.mean(news_embs, axis=0),
                'ctr': total_clicks / max(total_impressions, 1)
            })
    
    return sessions

day4_sessions = create_sessions(behaviors_train, news_embeddings)
day5_sessions = create_sessions(behaviors_test, news_embeddings)

print(f"✅ Sessions:")
print(f"   Day 4 training sessions: {len(day4_sessions):,}")
print(f"   Day 5 test sessions: {len(day5_sessions):,}")

if not day4_sessions or not day5_sessions:
    print("❌ No valid sessions created. Exiting.")
    exit(1)

# Convert to matrices
X_news_train = np.array([s['news_emb'] for s in day4_sessions])
y_train = np.array([s['ctr'] for s in day4_sessions])

X_news_test = np.array([s['news_emb'] for s in day5_sessions])
y_test = np.array([s['ctr'] for s in day5_sessions])

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

X_user_click_train = get_user_features(day4_sessions, day4_user_click_embs, user_emb_dim)
X_user_click_test = get_user_features(day5_sessions, day4_user_click_embs, user_emb_dim)

X_user_view_train = get_user_features(day4_sessions, day4_user_view_embs, user_emb_dim)
X_user_view_test = get_user_features(day5_sessions, day4_user_view_embs, user_emb_dim)

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

print(f"\n📊 Feature matrices (Day 4 → Day 5):")
print(f"   News: {X_news_train_norm.shape}")
print(f"   User (Click): {X_user_click_train_norm.shape}")
print(f"   User (View): {X_user_view_train_norm.shape}")
print(f"   Target CTR range: [{y_train.min():.3f}, {y_train.max():.3f}]")

print("\n🏆 TESTING RECENCY EFFECT: DAY 4 → DAY 5")
print("=" * 60)

results_recency = {}

# Test feature combinations
feature_combinations = [
    ('News_Only', X_news_train_norm, X_news_test_norm, 'News content only'),
    ('User_Click_Day4', X_user_click_train_norm, X_user_click_test_norm, 'Click-based user (Day 4)'),
    ('User_View_Day4', X_user_view_train_norm, X_user_view_test_norm, 'View-based user (Day 4)'),
    ('News_Click_Day4', np.hstack([X_news_train_norm, X_user_click_train_norm]), 
     np.hstack([X_news_test_norm, X_user_click_test_norm]), 'News + Click-based (Day 4)'),
    ('News_View_Day4', np.hstack([X_news_train_norm, X_user_view_train_norm]), 
     np.hstack([X_news_test_norm, X_user_view_test_norm]), 'News + View-based (Day 4)')
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
    
    results_recency[name] = {
        'MSE': mse,
        'R2': r2,
        'Correlation': corr,
        'Features': X_train.shape[1],
        'Description': description
    }
    
    print(f"   Ridge: R²={r2:.4f}, Corr={corr:.4f}, MSE={mse:.6f}")

print("\n📊 RECENCY EFFECT RESULTS: DAY 4 → DAY 5")
print("=" * 60)

df_recency = pd.DataFrame(results_recency).T
df_recency_sorted = df_recency.sort_values('R2', ascending=False)

print("🏆 PERFORMANCE RANKING (1-day gap):")
print("=" * 90)
print(df_recency_sorted[['R2', 'Correlation', 'MSE', 'Features', 'Description']].round(4).to_string())

print(f"\n🔍 RECENCY ANALYSIS:")
print("=" * 40)

click_day4_r2 = results_recency['User_Click_Day4']['R2']
view_day4_r2 = results_recency['User_View_Day4']['R2']
news_r2 = results_recency['News_Only']['R2']

click_combined_day4_r2 = results_recency['News_Click_Day4']['R2']
view_combined_day4_r2 = results_recency['News_View_Day4']['R2']

print(f"📈 RECENCY EFFECT (Day 4 → Day 5):")
print(f"   Click-based user (Day 4): R² = {click_day4_r2:.4f}")
print(f"   View-based user (Day 4):  R² = {view_day4_r2:.4f}")
print(f"   News only:                R² = {news_r2:.4f}")

print(f"\n📈 COMBINED FEATURES (Day 4 → Day 5):")
print(f"   News + Click-based: R² = {click_combined_day4_r2:.4f}")
print(f"   News + View-based:  R² = {view_combined_day4_r2:.4f}")

# Load previous results for comparison if available
try:
    df_previous = pd.read_csv('sampled_click_vs_view_results.csv', index_col=0)
    
    prev_click_r2 = df_previous.loc['User_Click_Only', 'R2']
    prev_view_r2 = df_previous.loc['User_View_Only', 'R2']
    prev_news_r2 = df_previous.loc['News_Only', 'R2']
    
    print(f"\n📊 COMPARISON: 1-DAY vs 4-DAY TEMPORAL GAP:")
    print("=" * 50)
    print(f"   Click-based user features:")
    print(f"     1-day gap (Day 4→5): R² = {click_day4_r2:.4f}")
    print(f"     4-day gap (Day 1-4→5): R² = {prev_click_r2:.4f}")
    print(f"     Recency improvement: {click_day4_r2 - prev_click_r2:+.4f}")
    
    print(f"\n   View-based user features:")
    print(f"     1-day gap (Day 4→5): R² = {view_day4_r2:.4f}")
    print(f"     4-day gap (Day 1-4→5): R² = {prev_view_r2:.4f}")
    print(f"     Recency improvement: {view_day4_r2 - prev_view_r2:+.4f}")
    
    print(f"\n   News-only features:")
    print(f"     1-day gap (Day 4→5): R² = {news_r2:.4f}")
    print(f"     4-day gap (Day 1-4→5): R² = {prev_news_r2:.4f}")
    print(f"     Recency improvement: {news_r2 - prev_news_r2:+.4f}")
    
except FileNotFoundError:
    print(f"\n📊 Previous results not found for comparison")

print(f"\n💡 RECENCY CONCLUSIONS:")
print("=" * 40)

if click_day4_r2 > view_day4_r2:
    print(f"🎉 BREAKTHROUGH: Click-based features now outperform view-based!")
    print(f"   Recency matters more than we thought")
elif abs(click_day4_r2 - view_day4_r2) < 0.05:
    print(f"📊 CLOSER PERFORMANCE: Recency reduces the gap")
else:
    print(f"📈 VIEW FEATURES STILL WIN: But gap may be smaller")

best_model = df_recency_sorted.iloc[0]
print(f"\n🥇 BEST RECENCY MODEL: {best_model.name}")
print(f"   R² = {best_model['R2']:.4f}")
print(f"   Strategy: {best_model['Description']}")

# Visualization
plt.figure(figsize=(12, 4))

# Performance comparison
plt.subplot(1, 2, 1)
models = ['News\nOnly', 'User\n(Click)', 'User\n(View)', 'News+Click', 'News+View']
r2_values = [results_recency['News_Only']['R2'], results_recency['User_Click_Day4']['R2'], 
             results_recency['User_View_Day4']['R2'], results_recency['News_Click_Day4']['R2'],
             results_recency['News_View_Day4']['R2']]

colors = ['blue', 'green', 'red', 'darkgreen', 'darkred']
bars = plt.bar(models, r2_values, color=colors, alpha=0.7)

plt.ylabel('R² Score')
plt.title('Recency Effect: Day 4 → Day 5')
plt.grid(True, alpha=0.3)

for bar, r2 in zip(bars, r2_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{r2:.3f}', ha='center', va='bottom', fontsize=9)

# User feature comparison
plt.subplot(1, 2, 2)
user_types = ['Click-based\n(Day 4)', 'View-based\n(Day 4)']
user_r2 = [click_day4_r2, view_day4_r2]
user_colors = ['green' if click_day4_r2 > view_day4_r2 else 'orange', 'red' if click_day4_r2 > view_day4_r2 else 'blue']

bars2 = plt.bar(user_types, user_r2, color=user_colors, alpha=0.7)
plt.ylabel('R² Score')
plt.title('User Features: 1-Day Temporal Gap')
plt.grid(True, alpha=0.3)

for bar, r2 in zip(bars2, user_r2):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{r2:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print(f"\n⏰ RECENCY ANALYSIS COMPLETE!")
print(f"   Temporal gap: 1 day (vs previous 4-day gap)")
print(f"   Key finding: {'Recency helps click-based features' if click_day4_r2 > view_day4_r2 else 'View-based features still superior even with recency'}")

# Save results
df_recency.to_csv('recency_day4_to_day5_results.csv')
print(f"💾 Results saved to recency_day4_to_day5_results.csv") 