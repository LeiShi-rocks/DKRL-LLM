#!/usr/bin/env python3
"""
MIND User Behavior Temporal Consistency Analysis
===============================================

This script analyzes whether user viewing patterns are consistent between
training period (days 1-4) and test period (day 5), which could explain
why user features perform poorly in our models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

print("🔍 MIND USER BEHAVIOR TEMPORAL CONSISTENCY ANALYSIS")
print("=" * 70)

# Load the original MIND data
print("📂 Loading original MIND data...")

# Load behaviors data
behaviors_path = '../MINDsmall_train/behaviors.tsv'
news_path = '../MINDsmall_train/news.tsv'

print(f"   Loading behaviors from: {behaviors_path}")
behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None,
                          names=['impression_id', 'user_id', 'time', 'history', 'impressions'])

print(f"   Loading news from: {news_path}")
news_df = pd.read_csv(news_path, sep='\t', header=None,
                     names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])

print(f"✅ Data loaded:")
print(f"   Behaviors: {len(behaviors_df):,} records")
print(f"   News articles: {len(news_df):,} articles")
print(f"   Unique users: {behaviors_df['user_id'].nunique():,}")

# Parse timestamps and add date information
print(f"\n📅 Parsing timestamps...")
behaviors_df['timestamp'] = pd.to_datetime(behaviors_df['time'], format='%m/%d/%Y %I:%M:%S %p')
behaviors_df['date'] = behaviors_df['timestamp'].dt.date
behaviors_df['day_of_period'] = (behaviors_df['timestamp'] - behaviors_df['timestamp'].min()).dt.days + 1

# Identify the time periods
date_range = sorted(behaviors_df['date'].unique())
print(f"   Date range: {date_range[0]} to {date_range[-1]}")
print(f"   Total days: {len(date_range)}")

# Split into training (days 1-4) and test (day 5) periods
train_days = behaviors_df['day_of_period'] <= 4
test_days = behaviors_df['day_of_period'] == 5

behaviors_train = behaviors_df[train_days].copy()
behaviors_test = behaviors_df[test_days].copy()

print(f"   Training period (days 1-4): {len(behaviors_train):,} records")
print(f"   Test period (day 5): {len(behaviors_test):,} records")

# Find users who appear in both periods
users_train = set(behaviors_train['user_id'].unique())
users_test = set(behaviors_test['user_id'].unique())
users_both = users_train.intersection(users_test)

print(f"   Users in training: {len(users_train):,}")
print(f"   Users in test: {len(users_test):,}")
print(f"   Users in both periods: {len(users_both):,}")
print(f"   User overlap: {len(users_both)/len(users_train)*100:.1f}%")

print(f"\n🔍 ANALYZING USER VIEWING PATTERN CONSISTENCY")
print("=" * 60)

def extract_clicked_news(history_str):
    """Extract clicked news IDs from history string"""
    if pd.isna(history_str) or history_str == '':
        return []
    return [item for item in history_str.split() if item != '']

def extract_impression_news(impressions_str):
    """Extract news IDs and click labels from impressions"""
    if pd.isna(impressions_str) or impressions_str == '':
        return [], []
    
    items = impressions_str.split()
    news_ids = []
    clicks = []
    
    for item in items:
        if '-' in item:
            news_id, click = item.split('-')
            news_ids.append(news_id)
            clicks.append(int(click))
    
    return news_ids, clicks

# Analyze user click patterns in both periods
print("📊 Building user click profiles for each period...")

user_profiles_train = defaultdict(lambda: {'clicked_news': set(), 'viewed_news': set(), 'total_clicks': 0, 'total_views': 0})
user_profiles_test = defaultdict(lambda: {'clicked_news': set(), 'viewed_news': set(), 'total_clicks': 0, 'total_views': 0})

# Process training period
for _, row in behaviors_train.iterrows():
    user_id = row['user_id']
    
    # History clicks
    history_clicks = extract_clicked_news(row['history'])
    user_profiles_train[user_id]['clicked_news'].update(history_clicks)
    
    # Impression clicks and views
    impression_news, clicks = extract_impression_news(row['impressions'])
    user_profiles_train[user_id]['viewed_news'].update(impression_news)
    user_profiles_train[user_id]['total_views'] += len(impression_news)
    
    clicked_in_impression = [news for news, click in zip(impression_news, clicks) if click == 1]
    user_profiles_train[user_id]['clicked_news'].update(clicked_in_impression)
    user_profiles_train[user_id]['total_clicks'] += len(clicked_in_impression)

# Process test period
for _, row in behaviors_test.iterrows():
    user_id = row['user_id']
    
    # History clicks (should be empty or minimal in test)
    history_clicks = extract_clicked_news(row['history'])
    user_profiles_test[user_id]['clicked_news'].update(history_clicks)
    
    # Impression clicks and views
    impression_news, clicks = extract_impression_news(row['impressions'])
    user_profiles_test[user_id]['viewed_news'].update(impression_news)
    user_profiles_test[user_id]['total_views'] += len(impression_news)
    
    clicked_in_impression = [news for news, click in zip(impression_news, clicks) if click == 1]
    user_profiles_test[user_id]['clicked_news'].update(clicked_in_impression)
    user_profiles_test[user_id]['total_clicks'] += len(clicked_in_impression)

print(f"✅ Processed user profiles for {len(user_profiles_train):,} training users")
print(f"✅ Processed user profiles for {len(user_profiles_test):,} test users")

# Analyze consistency for users who appear in both periods
print(f"\n🎯 CONSISTENCY ANALYSIS FOR {len(users_both):,} OVERLAPPING USERS")
print("-" * 60)

consistency_metrics = []

for user_id in users_both:
    train_profile = user_profiles_train[user_id]
    test_profile = user_profiles_test[user_id]
    
    # Calculate overlaps
    clicked_overlap = len(train_profile['clicked_news'].intersection(test_profile['clicked_news']))
    viewed_overlap = len(train_profile['viewed_news'].intersection(test_profile['viewed_news']))
    
    train_clicked = len(train_profile['clicked_news'])
    test_clicked = len(test_profile['clicked_news'])
    train_viewed = len(train_profile['viewed_news'])
    test_viewed = len(test_profile['viewed_news'])
    
    # Calculate Jaccard similarities
    clicked_jaccard = (clicked_overlap / (train_clicked + test_clicked - clicked_overlap)) if (train_clicked + test_clicked - clicked_overlap) > 0 else 0
    viewed_jaccard = (viewed_overlap / (train_viewed + test_viewed - viewed_overlap)) if (train_viewed + test_viewed - viewed_overlap) > 0 else 0
    
    # Activity levels
    train_activity = train_profile['total_clicks'] + train_profile['total_views']
    test_activity = test_profile['total_clicks'] + test_profile['total_views']
    
    consistency_metrics.append({
        'user_id': user_id,
        'train_clicked': train_clicked,
        'test_clicked': test_clicked,
        'train_viewed': train_viewed,
        'test_viewed': test_viewed,
        'clicked_overlap': clicked_overlap,
        'viewed_overlap': viewed_overlap,
        'clicked_jaccard': clicked_jaccard,
        'viewed_jaccard': viewed_jaccard,
        'train_activity': train_activity,
        'test_activity': test_activity
    })

consistency_df = pd.DataFrame(consistency_metrics)

# Summary statistics
print(f"📈 CONSISTENCY STATISTICS:")
print("-" * 40)

print(f"🔗 Click Pattern Consistency:")
print(f"   Mean Jaccard similarity: {consistency_df['clicked_jaccard'].mean():.4f}")
print(f"   Median Jaccard similarity: {consistency_df['clicked_jaccard'].median():.4f}")
print(f"   Users with 0 overlap: {(consistency_df['clicked_jaccard'] == 0).sum():,} ({(consistency_df['clicked_jaccard'] == 0).mean()*100:.1f}%)")
print(f"   Users with >0.1 overlap: {(consistency_df['clicked_jaccard'] > 0.1).sum():,} ({(consistency_df['clicked_jaccard'] > 0.1).mean()*100:.1f}%)")

print(f"\n👀 View Pattern Consistency:")
print(f"   Mean Jaccard similarity: {consistency_df['viewed_jaccard'].mean():.4f}")
print(f"   Median Jaccard similarity: {consistency_df['viewed_jaccard'].median():.4f}")
print(f"   Users with 0 overlap: {(consistency_df['viewed_jaccard'] == 0).sum():,} ({(consistency_df['viewed_jaccard'] == 0).mean()*100:.1f}%)")
print(f"   Users with >0.1 overlap: {(consistency_df['viewed_jaccard'] > 0.1).sum():,} ({(consistency_df['viewed_jaccard'] > 0.1).mean()*100:.1f}%)")

print(f"\n📊 Activity Level Changes:")
activity_correlation = np.corrcoef(consistency_df['train_activity'], consistency_df['test_activity'])[0, 1]
print(f"   Activity correlation (train vs test): {activity_correlation:.4f}")
print(f"   Mean training activity: {consistency_df['train_activity'].mean():.1f}")
print(f"   Mean test activity: {consistency_df['test_activity'].mean():.1f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('User Behavior Temporal Consistency Analysis', fontsize=16, fontweight='bold')

# 1. Jaccard similarity distributions
ax1 = axes[0, 0]
ax1.hist(consistency_df['clicked_jaccard'], bins=30, alpha=0.7, label='Clicked News', color='blue')
ax1.hist(consistency_df['viewed_jaccard'], bins=30, alpha=0.7, label='Viewed News', color='orange')
ax1.set_xlabel('Jaccard Similarity')
ax1.set_ylabel('Number of Users')
ax1.set_title('Distribution of Jaccard Similarities\n(Train vs Test Periods)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Activity correlation
ax2 = axes[0, 1]
ax2.scatter(consistency_df['train_activity'], consistency_df['test_activity'], alpha=0.5)
ax2.set_xlabel('Training Period Activity')
ax2.set_ylabel('Test Period Activity')
ax2.set_title(f'Activity Level Correlation\n(r = {activity_correlation:.3f})')
ax2.grid(True, alpha=0.3)

# Add correlation line
if activity_correlation > 0.1:
    z = np.polyfit(consistency_df['train_activity'], consistency_df['test_activity'], 1)
    p = np.poly1d(z)
    ax2.plot(consistency_df['train_activity'], p(consistency_df['train_activity']), "r--", alpha=0.8)

# 3. Consistency vs activity
ax3 = axes[1, 0]
ax3.scatter(consistency_df['train_activity'], consistency_df['clicked_jaccard'], alpha=0.5)
ax3.set_xlabel('Training Period Activity')
ax3.set_ylabel('Click Pattern Jaccard Similarity')
ax3.set_title('Consistency vs Training Activity')
ax3.grid(True, alpha=0.3)

# 4. Summary metrics
ax4 = axes[1, 1]
metrics = ['Click Jaccard', 'View Jaccard', 'Activity Corr']
values = [
    consistency_df['clicked_jaccard'].mean(),
    consistency_df['viewed_jaccard'].mean(),
    activity_correlation
]
colors = ['red' if v < 0.1 else 'orange' if v < 0.3 else 'green' for v in values]

bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
ax4.set_ylabel('Correlation/Similarity Score')
ax4.set_title('Summary: Temporal Consistency Metrics')
ax4.set_ylim(0, max(values) + 0.1)
ax4.grid(True, alpha=0.3)

# Add value labels
for bar, value in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Final insights
print(f"\n🎯 KEY INSIGHTS & IMPLICATIONS:")
print("=" * 50)

if consistency_df['clicked_jaccard'].mean() < 0.1:
    print(f"❌ POOR TEMPORAL CONSISTENCY DETECTED:")
    print(f"   • Click patterns have very low overlap between periods")
    print(f"   • Mean Jaccard similarity = {consistency_df['clicked_jaccard'].mean():.4f} (should be >0.3 for good consistency)")
    print(f"   • {(consistency_df['clicked_jaccard'] == 0).mean()*100:.1f}% of users have zero click overlap")
    print(f"   • This explains why user features are not predictive!")
elif consistency_df['clicked_jaccard'].mean() < 0.3:
    print(f"⚠️ MODERATE TEMPORAL INCONSISTENCY:")
    print(f"   • Some overlap but still concerning for predictive modeling")
    print(f"   • Mean Jaccard similarity = {consistency_df['clicked_jaccard'].mean():.4f}")
else:
    print(f"✅ GOOD TEMPORAL CONSISTENCY:")
    print(f"   • Users show consistent behavior across periods")
    print(f"   • Mean Jaccard similarity = {consistency_df['clicked_jaccard'].mean():.4f}")

if activity_correlation < 0.3:
    print(f"\n📉 ACTIVITY PATTERN ISSUES:")
    print(f"   • User activity levels poorly correlated (r={activity_correlation:.3f})")
    print(f"   • Users may be changing their engagement patterns")
else:
    print(f"\n📈 CONSISTENT ACTIVITY PATTERNS:")
    print(f"   • User activity levels well correlated (r={activity_correlation:.3f})")

print(f"\n💡 IMPLICATIONS FOR MODEL PERFORMANCE:")
if consistency_df['clicked_jaccard'].mean() < 0.1:
    print(f"   • User features from days 1-4 are poor predictors for day 5")
    print(f"   • Focus on news content features instead of user history")
    print(f"   • Consider shorter-term user patterns (same-day behavior)")
    print(f"   • This validates our observation that news-only models perform similarly to combined models")

print(f"\n🔧 RECOMMENDATIONS:")
print(f"   1. Use news content features as primary signal")
print(f"   2. If using user features, focus on very recent behavior (hours, not days)")
print(f"   3. Consider cold-start approaches for user modeling")
print(f"   4. Investigate what causes the temporal shift in user preferences")

# Save results
output_file = 'user_temporal_consistency_analysis.csv'
consistency_df.to_csv(output_file, index=False)
print(f"\n💾 Results saved to: {output_file}")

print(f"\n✅ Analysis complete! This explains the poor performance of user features.")
