#!/usr/bin/env python3
"""
Propensity Score Modeling for Recommendation Systems
===================================================

This module provides tools for learning and working with propensity scores
in recommendation systems to handle selection bias and confounding.

Key Features:
- Learn propensity scores from observed recommendation patterns
- Handle selection bias in click prediction
- Implement inverse propensity weighting (IPW)
- Position bias modeling
- Popularity bias correction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
import warnings
warnings.filterwarnings('ignore')

class PropensityScoreModel:
    """
    Learn propensity scores for recommendation systems
    
    The propensity score P(T=1|X,Z) is the probability that item Z
    is recommended to user X, given their features.
    """
    
    def __init__(self, model_type='logistic', **kwargs):
        """
        Initialize propensity score model
        
        Args:
            model_type: 'logistic' or 'forest'
            **kwargs: Parameters for underlying model
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'logistic':
            self.model = LogisticRegression(**kwargs)
        elif model_type == 'forest':
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError("model_type must be 'logistic' or 'forest'")
        
        self.is_fitted = False
    
    def _create_features(self, X_user, X_news, positions=None, include_interactions=True):
        """Create feature matrix for propensity modeling"""
        n_samples = len(X_user)
        
        # Basic features
        features = np.hstack([X_user, X_news])
        
        # Position features
        if positions is not None:
            pos_features = np.column_stack([
                positions,                    # Raw position
                1.0 / (1.0 + positions),     # Position bias term
                np.log(1.0 + positions)      # Log position
            ])
            features = np.hstack([features, pos_features])
        
        # Interaction features (user × news)
        if include_interactions and X_user.shape[1] <= 10 and X_news.shape[1] <= 10:
            # Only compute interactions for small feature sets
            interactions = []
            for i in range(X_user.shape[1]):
                for j in range(X_news.shape[1]):
                    interactions.append(X_user[:, i] * X_news[:, j])
            
            if interactions:
                interaction_matrix = np.column_stack(interactions)
                features = np.hstack([features, interaction_matrix])
        
        return features
    
    def fit(self, X_user, X_news, recommended, positions=None, sample_weight=None):
        """
        Fit propensity score model
        
        Args:
            X_user: User features (n_samples, n_user_features)
            X_news: News features (n_samples, n_news_features)  
            recommended: Binary indicator if item was recommended (n_samples,)
            positions: Position in recommendation list (optional)
            sample_weight: Sample weights (optional)
        """
        
        # Create feature matrix
        X = self._create_features(X_user, X_news, positions)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        if sample_weight is not None:
            self.model.fit(X_scaled, recommended, sample_weight=sample_weight)
        else:
            self.model.fit(X_scaled, recommended)
        
        self.is_fitted = True
        
        # Store training performance
        train_proba = self.model.predict_proba(X_scaled)[:, 1]
        self.train_auc = roc_auc_score(recommended, train_proba)
        self.train_logloss = log_loss(recommended, train_proba)
        
        return self
    
    def predict_proba(self, X_user, X_news, positions=None):
        """Predict propensity scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._create_features(X_user, X_news, positions)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X_user, X_news, positions=None, threshold=0.5):
        """Predict binary recommendations"""
        proba = self.predict_proba(X_user, X_news, positions)
        return (proba >= threshold).astype(int)


class InversePropensityWeighting:
    """
    Inverse Propensity Weighting for handling selection bias
    
    Uses propensity scores to reweight observations and reduce
    selection bias in click prediction models.
    """
    
    def __init__(self, propensity_model=None, clip_weights=(0.01, 10.0)):
        """
        Initialize IPW
        
        Args:
            propensity_model: Fitted PropensityScoreModel
            clip_weights: (min, max) values for weight clipping
        """
        self.propensity_model = propensity_model
        self.clip_min, self.clip_max = clip_weights
    
    def compute_weights(self, X_user, X_news, recommended, positions=None):
        """
        Compute inverse propensity weights
        
        Weight = 1 / P(recommended=1 | user, news, position)
        Only computed for actually recommended items.
        """
        
        if self.propensity_model is None:
            raise ValueError("Must provide fitted propensity model")
        
        # Get propensity scores
        propensities = self.propensity_model.predict_proba(X_user, X_news, positions)
        
        # Compute weights only for recommended items
        weights = np.zeros(len(recommended))
        recommended_mask = recommended == 1
        
        if np.sum(recommended_mask) > 0:
            # IPW formula: 1 / P(T=1|X,Z) 
            weights[recommended_mask] = 1.0 / propensities[recommended_mask]
            
            # Clip extreme weights
            weights = np.clip(weights, self.clip_min, self.clip_max)
        
        return weights
    
    def effective_sample_size(self, weights):
        """Compute effective sample size after weighting"""
        if np.sum(weights) == 0:
            return 0
        return (np.sum(weights) ** 2) / np.sum(weights ** 2)


class PositionBiasModel:
    """
    Model position bias in recommendation systems
    
    P(click | position) typically decreases with position due to
    user attention patterns.
    """
    
    def __init__(self, model_type='exponential'):
        """
        Initialize position bias model
        
        Args:
            model_type: 'exponential', 'power', or 'logistic'
        """
        self.model_type = model_type
        self.is_fitted = False
    
    def fit(self, positions, clicks):
        """
        Fit position bias model
        
        Args:
            positions: Position in recommendation list (1-indexed)
            clicks: Binary click indicators
        """
        
        positions = np.array(positions)
        clicks = np.array(clicks)
        
        # Aggregate by position
        pos_stats = {}
        for pos in np.unique(positions):
            pos_mask = positions == pos
            pos_stats[pos] = {
                'clicks': np.sum(clicks[pos_mask]),
                'impressions': np.sum(pos_mask),
                'ctr': np.mean(clicks[pos_mask])
            }
        
        self.position_stats = pos_stats
        
        # Fit parametric model
        positions_unique = sorted(pos_stats.keys())
        ctrs = [pos_stats[pos]['ctr'] for pos in positions_unique]
        
        if self.model_type == 'exponential':
            # CTR = a * exp(-b * position)
            log_ctrs = np.log(np.maximum(ctrs, 1e-6))
            poly_coef = np.polyfit(positions_unique, log_ctrs, 1)
            self.params = {'a': np.exp(poly_coef[1]), 'b': -poly_coef[0]}
            
        elif self.model_type == 'power':
            # CTR = a * position^(-b)
            log_pos = np.log(positions_unique)
            log_ctrs = np.log(np.maximum(ctrs, 1e-6))
            poly_coef = np.polyfit(log_pos, log_ctrs, 1)
            self.params = {'a': np.exp(poly_coef[1]), 'b': -poly_coef[0]}
            
        elif self.model_type == 'logistic':
            # Fit logistic regression
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression()
            X = np.array(positions_unique).reshape(-1, 1)
            
            # Create weighted samples
            X_expanded = []
            y_expanded = []
            for pos in positions_unique:
                stats = pos_stats[pos]
                X_expanded.extend([pos] * stats['impressions'])
                y_expanded.extend([1] * stats['clicks'] + [0] * (stats['impressions'] - stats['clicks']))
            
            lr.fit(np.array(X_expanded).reshape(-1, 1), y_expanded)
            self.params = {'model': lr}
        
        self.is_fitted = True
        return self
    
    def predict_bias(self, positions):
        """Predict position bias (probability of click due to position)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        positions = np.array(positions)
        
        if self.model_type == 'exponential':
            return self.params['a'] * np.exp(-self.params['b'] * positions)
            
        elif self.model_type == 'power':
            return self.params['a'] * (positions ** (-self.params['b']))
            
        elif self.model_type == 'logistic':
            return self.params['model'].predict_proba(positions.reshape(-1, 1))[:, 1]
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


def doubly_robust_estimator(y_obs, recommended, propensity_scores, outcome_model_pred):
    """
    Doubly robust estimator for causal effect estimation
    
    Combines outcome modeling with inverse propensity weighting
    for more robust causal inference.
    
    Args:
        y_obs: Observed outcomes (only for recommended items)
        recommended: Binary treatment indicator  
        propensity_scores: Estimated propensity scores
        outcome_model_pred: Outcome model predictions
    
    Returns:
        Doubly robust estimate of average treatment effect
    """
    
    # Only use recommended items (observed outcomes)
    mask = recommended == 1
    y_obs_rec = y_obs[mask]
    prop_rec = propensity_scores[mask]
    outcome_pred_rec = outcome_model_pred[mask]
    
    # IPW term
    ipw_term = y_obs_rec / prop_rec
    
    # Bias correction term
    bias_correction = (1 - 1/prop_rec) * outcome_pred_rec
    
    # Doubly robust estimator
    dr_estimates = ipw_term + bias_correction
    
    return {
        'dr_estimate': np.mean(dr_estimates),
        'dr_std': np.std(dr_estimates) / np.sqrt(len(dr_estimates)),
        'ipw_estimate': np.mean(ipw_term),
        'outcome_model_estimate': np.mean(outcome_pred_rec)
    }


# Example usage and testing functions
def demonstrate_propensity_modeling():
    """Demonstrate propensity score modeling with synthetic data"""
    
    print("🎯 Propensity Score Modeling Demonstration")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # User and news features
    X_user = np.random.normal(0, 1, (n_samples, 5))
    X_news = np.random.normal(0, 1, (n_samples, 3))
    positions = np.random.randint(1, 11, n_samples)
    
    # True propensity model (unknown in practice)
    true_logits = (
        2.0 * (X_user[:, 0] * X_news[:, 0]) +  # User-news interaction
        1.0 * X_news[:, 1] +                   # News quality
        -0.5 * np.log(positions)               # Position bias
    )
    true_propensities = 1.0 / (1.0 + np.exp(-true_logits))
    recommended = np.random.binomial(1, true_propensities)
    
    print(f"   Generated {n_samples:,} synthetic observations")
    print(f"   Recommendation rate: {recommended.mean():.3f}")
    
    # Fit propensity model
    prop_model = PropensityScoreModel(model_type='logistic')
    prop_model.fit(X_user, X_news, recommended, positions)
    
    print(f"   Fitted propensity model (AUC: {prop_model.train_auc:.3f})")
    
    # Predict propensities
    pred_propensities = prop_model.predict_proba(X_user, X_news, positions)
    correlation = np.corrcoef(true_propensities, pred_propensities)[0, 1]
    
    print(f"   Correlation with true propensities: {correlation:.3f}")
    
    # Demonstrate IPW
    ipw = InversePropensityWeighting(prop_model)
    weights = ipw.compute_weights(X_user, X_news, recommended, positions)
    
    print(f"   IPW weights - Mean: {weights[recommended==1].mean():.2f}, Max: {weights.max():.2f}")
    print(f"   Effective sample size: {ipw.effective_sample_size(weights):.1f}")
    
    return prop_model, ipw


if __name__ == "__main__":
    demonstrate_propensity_modeling()

