#!/usr/bin/env python3
"""
MIND Two-Stage Ensemble: Ridge + DKRL Residual Modeling
======================================================

Strategy:
1. Fit Ridge regression on combined user + news features
2. Calculate residuals from Ridge predictions
3. Apply DKRL to residuals to capture additional interaction patterns
4. Final prediction = Ridge prediction + DKRL residual prediction

This tests whether DKRL can add value as a second-stage model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import warnings
warnings.filterwarnings('ignore')

print("🎯 MIND TWO-STAGE ENSEMBLE: RIDGE + DKRL RESIDUAL MODELING")
print("=" * 70)

# Load preprocessed data
print("📂 Loading preprocessed MIND data...")
data_path = '../data_processing/'

try:
    X_news = np.load(data_path + 'X_news_embeddings.npy')
    X_user = np.load(data_path + 'X_user_features.npy') 
    y = np.load(data_path + 'y_clicks.npy')
    
    print(f"✅ Data loaded successfully:")
    print(f"   News features: {X_news.shape}")
    print(f"   User features: {X_user.shape}")
    print(f"   Target: {y.shape}")
    print(f"   Target range: [{y.min():.3f}, {y.max():.3f}]")
    
except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}")
    print("   Make sure the data_processing folder contains the required .npy files")
    exit(1)

# Check data consistency
assert X_news.shape[0] == X_user.shape[0] == len(y), "Data shapes must match"

# SUBSAMPLE FOR FASTER TRAINING
print(f"\n🎲 Subsampling for faster training...")
sample_size = 5000
np.random.seed(42)

if len(y) > sample_size:
    sample_indices = np.random.choice(len(y), sample_size, replace=False)
    X_news = X_news[sample_indices]
    X_user = X_user[sample_indices]
    y = y[sample_indices]
    print(f"   Sampled {sample_size:,} from {len(sample_indices):,} total samples")
else:
    print(f"   Using all {len(y):,} samples (dataset smaller than requested sample size)")

# Basic data info
print(f"\n📊 Dataset Statistics (after sampling):")
print(f"   Total samples: {len(y):,}")
print(f"   News embedding dim: {X_news.shape[1]}")
print(f"   User feature dim: {X_user.shape[1]}")
print(f"   Target mean: {y.mean():.4f}")
print(f"   Target std: {y.std():.4f}")

# Handle binary vs continuous target
is_binary = len(np.unique(y)) == 2
print(f"   Target type: {'Binary' if is_binary else 'Continuous'}")

if is_binary:
    print(f"   Class distribution: {np.bincount(y.astype(int))}")
else:
    print(f"   Target quartiles: {np.percentile(y, [25, 50, 75])}")

# Combine features for Ridge regression
print("\n🔗 Combining user and news features...")
X_combined = np.hstack([X_user, X_news])
print(f"   Combined features shape: {X_combined.shape}")

# Train/test split
print("\n📊 Creating train/test split...")
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=test_size, random_state=42, stratify=y if is_binary else None
)

X_user_train, X_user_test = train_test_split(
    X_user, test_size=test_size, random_state=42, stratify=y if is_binary else None
)

X_news_train, X_news_test = train_test_split(
    X_news, test_size=test_size, random_state=42, stratify=y if is_binary else None
)

print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")

# Normalize features
print("\n🔧 Normalizing features...")
scaler_combined = StandardScaler()
scaler_user = StandardScaler()
scaler_news = StandardScaler()

X_train_norm = scaler_combined.fit_transform(X_train)
X_test_norm = scaler_combined.transform(X_test)

X_user_train_norm = scaler_user.fit_transform(X_user_train)
X_user_test_norm = scaler_user.transform(X_user_test)

X_news_train_norm = scaler_news.fit_transform(X_news_train)
X_news_test_norm = scaler_news.transform(X_news_test)

print("✅ Feature normalization completed")

print("\n🏆 STAGE 1: RIDGE REGRESSION BASELINE")
print("=" * 50)

# Train Ridge regression on combined features
print("📊 Training Ridge regression on combined features...")
start_time = time.time()

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_norm, y_train)

# Get Ridge predictions
y_train_ridge = ridge.predict(X_train_norm)
y_test_ridge = ridge.predict(X_test_norm)

ridge_time = time.time() - start_time

# Calculate Ridge performance
ridge_train_mse = mean_squared_error(y_train, y_train_ridge)
ridge_test_mse = mean_squared_error(y_test, y_test_ridge)
ridge_train_r2 = r2_score(y_train, y_train_ridge)
ridge_test_r2 = r2_score(y_test, y_test_ridge)
ridge_train_corr = np.corrcoef(y_train, y_train_ridge)[0, 1]
ridge_test_corr = np.corrcoef(y_test, y_test_ridge)[0, 1]

print(f"✅ Ridge regression completed in {ridge_time:.2f}s")
print(f"   Training - MSE: {ridge_train_mse:.6f}, R²: {ridge_train_r2:.4f}, Corr: {ridge_train_corr:.4f}")
print(f"   Test     - MSE: {ridge_test_mse:.6f}, R²: {ridge_test_r2:.4f}, Corr: {ridge_test_corr:.4f}")

# Calculate residuals for DKRL
print("\n🔍 Calculating residuals for DKRL...")
residuals_train = y_train - y_train_ridge
residuals_test = y_test - y_test_ridge

print(f"   Training residuals - Mean: {residuals_train.mean():.6f}, Std: {residuals_train.std():.6f}")
print(f"   Test residuals     - Mean: {residuals_test.mean():.6f}, Std: {residuals_test.std():.6f}")

print("\n🧠 STAGE 2: DKRL ON RESIDUALS")
print("=" * 50)

def DKRL_residual_model(Z, X, y_residual, r, penalty, tol=1e-4, T=1000, verbose=False):
    """
    DKRL model for residual learning
    
    Args:
        Z: News features (n_samples x n_news_features)
        X: User features (n_samples x n_user_features) 
        y_residual: Residuals from Ridge regression
        r: Rank parameter
        penalty: Regularization parameter
    """
    # Ensure float64 for numerical stability
    Z = Z.astype(np.float64)
    X = X.astype(np.float64)
    y_residual = y_residual.astype(np.float64)
    penalty = float(penalty)
    
    p, q, N = Z.shape[1], X.shape[1], len(y_residual)
    
    # Initialize matrices with small random values
    np.random.seed(42)
    U = np.random.normal(0, 0.01, size=(p, r)).astype(np.float64)
    V = np.random.normal(0, 0.01, size=(q, r)).astype(np.float64)
    
    best_obj = np.inf
    best_U, best_V = U.copy(), V.copy()
    
    if verbose:
        print(f"   DKRL training: rank={r}, penalty={penalty:.4f}, samples={N}")
    
    for iteration in range(T):
        # Update U (news feature matrix)
        H = X @ V  # N x r
        for i in range(r):
            # Compute residual for this component
            y_temp = y_residual.copy()
            for j in range(r):
                if j != i:
                    y_temp -= (Z @ U[:, j]) * (X @ V[:, j])
            
            # Solve weighted least squares for U[:, i]
            A = Z.T @ np.diag(H[:, i]**2 + 1e-8) @ Z + penalty * np.eye(p)
            b = Z.T @ (H[:, i] * y_temp)
            
            # Add regularization if ill-conditioned
            if np.linalg.cond(A) > 1e12:
                A += penalty * np.eye(p)
            
            try:
                U[:, i] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                U[:, i] = np.linalg.pinv(A) @ b
        
        # Update V (user feature matrix)
        G = Z @ U  # N x r
        for j in range(r):
            # Compute residual for this component
            y_temp = y_residual.copy()
            for i in range(r):
                if i != j:
                    y_temp -= (Z @ U[:, i]) * (X @ V[:, i])
            
            # Solve weighted least squares for V[:, j]
            A = X.T @ np.diag(G[:, j]**2 + 1e-8) @ X + penalty * np.eye(q)
            b = X.T @ (G[:, j] * y_temp)
            
            if np.linalg.cond(A) > 1e12:
                A += penalty * np.eye(q)
            
            try:
                V[:, j] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                V[:, j] = np.linalg.pinv(A) @ b
        
        # Check convergence
        if iteration % 100 == 0:
            # Compute current predictions
            G = Z @ U
            H = X @ V
            y_pred = np.sum(G * H, axis=1)
            
            # Compute objective
            mse_loss = np.mean((y_residual - y_pred) ** 2)
            reg_loss = 0.5 * penalty * (np.sum(U**2) + np.sum(V**2))
            obj = mse_loss + reg_loss
            
            if obj < best_obj:
                best_obj = obj
                best_U, best_V = U.copy(), V.copy()
            
            if verbose and iteration % 200 == 0:
                r2_temp = r2_score(y_residual, y_pred)
                print(f"     Iteration {iteration}: Obj={obj:.6f}, R²={r2_temp:.4f}")
            
            # Early stopping
            if iteration > 200 and abs(obj - best_obj) < tol:
                if verbose:
                    print(f"     Converged at iteration {iteration}")
                break
    
    return best_U, best_V

# Hyperparameter tuning for DKRL
print("🔧 DKRL hyperparameter tuning on residuals...")

r_values = [2, 3, 5, 10]
penalty_values = [0.01, 0.1, 1.0, 10.0]

best_dkrl_r2 = -np.inf
best_dkrl_params = {}
dkrl_results = []

# Use subset for fast tuning
n_tune = min(800, len(X_train_norm))  # Larger subset for 5000 samples
tune_indices = np.random.choice(len(X_train_norm), n_tune, replace=False)

X_news_tune = X_news_train_norm[tune_indices]
X_user_tune = X_user_train_norm[tune_indices]
residuals_tune = residuals_train[tune_indices]

print(f"   Tuning on {n_tune} samples...")

for r in r_values:
    for penalty in penalty_values:
        print(f"   Testing r={r}, penalty={penalty:.2f}")
        
        try:
            start_time = time.time()
            U, V = DKRL_residual_model(
                X_news_tune, X_user_tune, residuals_tune, 
                r=r, penalty=penalty, tol=1e-4, T=500, verbose=False
            )
            
            # Predict residuals
            G = X_news_tune @ U
            H = X_user_tune @ V
            residual_pred = np.sum(G * H, axis=1)
            
            # Evaluate residual prediction
            r2_residual = r2_score(residuals_tune, residual_pred)
            mse_residual = mean_squared_error(residuals_tune, residual_pred)
            corr_residual = np.corrcoef(residuals_tune, residual_pred)[0, 1]
            
            tune_time = time.time() - start_time
            
            dkrl_results.append({
                'r': r, 'penalty': penalty, 'r2_residual': r2_residual, 
                'mse_residual': mse_residual, 'corr_residual': corr_residual, 'time': tune_time
            })
            
            if r2_residual > best_dkrl_r2:
                best_dkrl_r2 = r2_residual
                best_dkrl_params = {'r': r, 'penalty': penalty}
                print(f"     ✓ New best: R²={r2_residual:.4f}, MSE={mse_residual:.6f}")
            else:
                print(f"     R²={r2_residual:.4f}, MSE={mse_residual:.6f}")
                
        except Exception as e:
            print(f"     ❌ Failed: {str(e)[:50]}...")
            dkrl_results.append({
                'r': r, 'penalty': penalty, 'r2_residual': -999, 
                'mse_residual': 999, 'corr_residual': 0, 'time': 0
            })

print(f"\n🏆 Best DKRL parameters for residuals:")
print(f"   Rank: {best_dkrl_params.get('r', 'None')}")
print(f"   Penalty: {best_dkrl_params.get('penalty', 'None')}")
print(f"   Residual R²: {best_dkrl_r2:.4f}")

# Train final DKRL model on full training set
if best_dkrl_params and best_dkrl_r2 > -10:
    print(f"\n🚀 Training final DKRL model on full training set...")
    
    start_time = time.time()
    U_final, V_final = DKRL_residual_model(
        X_news_train_norm, X_user_train_norm, residuals_train,
        r=best_dkrl_params['r'], penalty=best_dkrl_params['penalty'],
        tol=1e-4, T=1000, verbose=True
    )
    dkrl_train_time = time.time() - start_time
    
    # Get DKRL residual predictions
    G_train = X_news_train_norm @ U_final
    H_train = X_user_train_norm @ V_final
    residual_pred_train = np.sum(G_train * H_train, axis=1)
    
    G_test = X_news_test_norm @ U_final
    H_test = X_user_test_norm @ V_final
    residual_pred_test = np.sum(G_test * H_test, axis=1)
    
    print(f"✅ DKRL training completed in {dkrl_train_time:.2f}s")
    
    print("\n🎯 STAGE 3: ENSEMBLE PREDICTIONS")
    print("=" * 50)
    
    # Ensemble predictions: Ridge + DKRL residuals
    y_train_ensemble = y_train_ridge + residual_pred_train
    y_test_ensemble = y_test_ridge + residual_pred_test
    
    # Calculate ensemble performance
    ensemble_train_mse = mean_squared_error(y_train, y_train_ensemble)
    ensemble_test_mse = mean_squared_error(y_test, y_test_ensemble)
    ensemble_train_r2 = r2_score(y_train, y_train_ensemble)
    ensemble_test_r2 = r2_score(y_test, y_test_ensemble)
    ensemble_train_corr = np.corrcoef(y_train, y_train_ensemble)[0, 1]
    ensemble_test_corr = np.corrcoef(y_test, y_test_ensemble)[0, 1]
    
    print(f"✅ Ensemble predictions completed")
    print(f"   Training - MSE: {ensemble_train_mse:.6f}, R²: {ensemble_train_r2:.4f}, Corr: {ensemble_train_corr:.4f}")
    print(f"   Test     - MSE: {ensemble_test_mse:.6f}, R²: {ensemble_test_r2:.4f}, Corr: {ensemble_test_corr:.4f}")
    
    # DKRL residual performance
    residual_train_r2 = r2_score(residuals_train, residual_pred_train)
    residual_test_r2 = r2_score(residuals_test, residual_pred_test)
    residual_train_corr = np.corrcoef(residuals_train, residual_pred_train)[0, 1]
    residual_test_corr = np.corrcoef(residuals_test, residual_pred_test)[0, 1]
    
    print(f"\n📊 DKRL Residual Performance:")
    print(f"   Training - R²: {residual_train_r2:.4f}, Corr: {residual_train_corr:.4f}")
    print(f"   Test     - R²: {residual_test_r2:.4f}, Corr: {residual_test_corr:.4f}")
    
else:
    print("❌ DKRL hyperparameter tuning failed - skipping ensemble")
    y_train_ensemble = y_train_ridge
    y_test_ensemble = y_test_ridge
    ensemble_train_mse, ensemble_test_mse = ridge_train_mse, ridge_test_mse
    ensemble_train_r2, ensemble_test_r2 = ridge_train_r2, ridge_test_r2
    ensemble_train_corr, ensemble_test_corr = ridge_train_corr, ridge_test_corr

print("\n📊 FINAL COMPARISON: RIDGE vs ENSEMBLE")
print("=" * 70)

# Results comparison
results = {
    'Ridge_Baseline': {
        'Train_MSE': ridge_train_mse, 'Test_MSE': ridge_test_mse,
        'Train_R2': ridge_train_r2, 'Test_R2': ridge_test_r2,
        'Train_Corr': ridge_train_corr, 'Test_Corr': ridge_test_corr,
        'Training_Time': ridge_time
    },
    'Ridge_DKRL_Ensemble': {
        'Train_MSE': ensemble_train_mse, 'Test_MSE': ensemble_test_mse,
        'Train_R2': ensemble_train_r2, 'Test_R2': ensemble_test_r2,
        'Train_Corr': ensemble_train_corr, 'Test_Corr': ensemble_test_corr,
        'Training_Time': ridge_time + (dkrl_train_time if 'dkrl_train_time' in locals() else 0)
    }
}

df_results = pd.DataFrame(results).T
print("🏆 PERFORMANCE COMPARISON:")
print("=" * 80)
print(df_results.round(6).to_string())

# Calculate improvements
test_r2_improvement = ensemble_test_r2 - ridge_test_r2
test_mse_improvement = ridge_test_mse - ensemble_test_mse
test_corr_improvement = ensemble_test_corr - ridge_test_corr

print(f"\n📈 ENSEMBLE IMPROVEMENTS:")
print(f"   Test R² improvement: {test_r2_improvement:+.6f}")
print(f"   Test MSE improvement: {test_mse_improvement:+.6f}")
print(f"   Test Correlation improvement: {test_corr_improvement:+.6f}")

print(f"\n💡 ANALYSIS:")
if test_r2_improvement > 0.01:
    print(f"✅ SIGNIFICANT IMPROVEMENT: DKRL adds value to Ridge regression!")
    print(f"   The ensemble approach successfully captures additional patterns")
elif test_r2_improvement > 0.001:
    print(f"📊 MODEST IMPROVEMENT: DKRL provides small but measurable benefit")
elif abs(test_r2_improvement) < 0.001:
    print(f"📊 NO SIGNIFICANT CHANGE: DKRL doesn't add substantial value")
else:
    print(f"❌ PERFORMANCE DEGRADATION: DKRL may be overfitting residuals")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Ridge + DKRL Ensemble Analysis', fontsize=16, fontweight='bold')

# 1. Performance comparison
ax1 = axes[0, 0]
models = ['Ridge\nBaseline', 'Ridge+DKRL\nEnsemble']
test_r2_values = [ridge_test_r2, ensemble_test_r2]
colors = ['blue', 'green' if test_r2_improvement > 0 else 'red']

bars = ax1.bar(models, test_r2_values, color=colors, alpha=0.7)
ax1.set_ylabel('Test R² Score')
ax1.set_title('Model Performance Comparison')
ax1.grid(True, alpha=0.3)

for bar, r2 in zip(bars, test_r2_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{r2:.4f}', ha='center', va='bottom', fontweight='bold')

# 2. Residual analysis (if DKRL was trained)
ax2 = axes[0, 1]
if 'residual_pred_test' in locals():
    ax2.scatter(residuals_test, residual_pred_test, alpha=0.5)
    ax2.plot([residuals_test.min(), residuals_test.max()], 
             [residuals_test.min(), residuals_test.max()], 'r--', alpha=0.8)
    ax2.set_xlabel('True Residuals')
    ax2.set_ylabel('DKRL Predicted Residuals')
    ax2.set_title(f'DKRL Residual Predictions\n(R² = {residual_test_r2:.4f})')
else:
    ax2.text(0.5, 0.5, 'DKRL training failed', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Residual Predictions')
ax2.grid(True, alpha=0.3)

# 3. Prediction comparison
ax3 = axes[1, 0]
ax3.scatter(y_test, y_test_ridge, alpha=0.5, label=f'Ridge (R²={ridge_test_r2:.3f})', color='blue')
ax3.scatter(y_test, y_test_ensemble, alpha=0.5, label=f'Ensemble (R²={ensemble_test_r2:.3f})', color='green')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8)
ax3.set_xlabel('True Values')
ax3.set_ylabel('Predicted Values')
ax3.set_title('Prediction Quality Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Improvement breakdown
ax4 = axes[1, 1]
metrics = ['R²', 'Correlation', 'MSE']
improvements = [test_r2_improvement, test_corr_improvement, -test_mse_improvement/ridge_test_mse]  # Normalize MSE
colors_improv = ['green' if imp > 0 else 'red' for imp in improvements]

bars = ax4.bar(metrics, improvements, color=colors_improv, alpha=0.7)
ax4.set_ylabel('Improvement')
ax4.set_title('Ensemble vs Ridge Improvements')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.grid(True, alpha=0.3)

for bar, imp in zip(bars, improvements):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001 if imp > 0 else bar.get_height() - 0.001,
             f'{imp:+.4f}', ha='center', va='bottom' if imp > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.show()

print(f"\n🎊 TWO-STAGE ENSEMBLE ANALYSIS COMPLETE!")
print(f"   Strategy: Ridge regression + DKRL residual modeling")
print(f"   Outcome: {'DKRL adds value' if test_r2_improvement > 0.001 else 'No significant benefit from DKRL'}")

# Save results
df_results.to_csv('ridge_dkrl_ensemble_results.csv')
if dkrl_results:
    pd.DataFrame(dkrl_results).to_csv('dkrl_hyperparameter_results.csv')
print(f"💾 Results saved!")
