"""
Texas Flood Detection - Utility Functions

This module contains reusable functions for flood detection analysis,
feature engineering, and anomaly detection using KDE and other methods.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def add_temporal_features(df):
    """
    Add rolling window features to state-day precipitation data.
    Features are computed per-state to avoid cross-state contamination.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns: 'state', 'date', 'precipitation', 'is_flood', 'flood_count'
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added temporal rolling window features (3, 7, 14 days)
    """
    df = df.sort_values(['state', 'date']).copy()
    
    # Rolling windows (Fourier-justified: 3, 7, 14 days)
    windows = [3, 7, 14]
    
    feature_dfs = []
    
    for state in df['state'].unique():
        state_df = df[df['state'] == state].copy()
        state_df = state_df.set_index('date')
        
        # Ensure continuous date index (fill missing dates with 0)
        date_range = pd.date_range(start=state_df.index.min(), end=state_df.index.max(), freq='D')
        state_df = state_df.reindex(date_range)
        state_df['state'] = state
        state_df['precipitation'] = state_df['precipitation'].fillna(0)
        state_df['is_flood'] = state_df['is_flood'].fillna(0)
        state_df['flood_count'] = state_df['flood_count'].fillna(0)
        
        precip = state_df['precipitation']
        
        for w in windows:
            # Cumulative precipitation over window (sum of past w days including today)
            state_df[f'precip_{w}d_sum'] = precip.rolling(window=w, min_periods=1).sum()
            
            # Maximum precipitation in window
            state_df[f'precip_{w}d_max'] = precip.rolling(window=w, min_periods=1).max()
            
            # Standard deviation (variability)
            state_df[f'precip_{w}d_std'] = precip.rolling(window=w, min_periods=1).std().fillna(0)
            
            # Mean precipitation
            state_df[f'precip_{w}d_mean'] = precip.rolling(window=w, min_periods=1).mean()
        
            # Rate of change: difference between x-day mean now vs x days ago
            state_df[f'precip_change_{w}d'] = state_df[f'precip_{w}d_mean'].diff(w).fillna(0)
        
        # Reset index to get date as column
        state_df = state_df.reset_index().rename(columns={'index': 'date'})
        feature_dfs.append(state_df)
    
    return pd.concat(feature_dfs, ignore_index=True)


def add_regional_features(df):
    """
    Add regional context features - precipitation and flood patterns across neighboring states.
    IMPORTANT: All features use PAST information only to avoid data leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns: 'state', 'date', 'precipitation', 'is_flood'
    region_states : list
        List of state abbreviations to consider in regional aggregates
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added regional features:
        - regional_precip_total: Total precipitation across all region states on given date
        - regional_precip_mean: Mean precipitation across region
        - regional_precip_7d: 7-day rolling sum of regional precipitation
        - regional_n_states_rain: Number of states with precipitation on given date
        - state_vs_regional_ratio: Ratio of this state's precip to regional average
        - recent_flood_history: Whether this state had floods in past 3 days (shifted to avoid leakage)
    """
    df = df.sort_values(['state', 'date']).copy()
    
    # First, compute regional daily aggregates (current day precip only - not floods to avoid leakage)
    regional_daily = df.groupby('date').agg(
        regional_precip_total=('precipitation', 'sum'),
        regional_precip_mean=('precipitation', 'mean'),
        regional_n_states_rain=('precipitation', lambda x: (x > 0).sum()),
    ).reset_index()
    
    # Add rolling regional features using ONLY past data
    regional_daily = regional_daily.sort_values('date')
    regional_daily['regional_precip_7d'] = regional_daily['regional_precip_total'].rolling(7, min_periods=1).sum()
    
    # Merge back to state-level data
    df = pd.merge(df, regional_daily, on='date', how='left')
    
    # Compute state-vs-regional ratio (how intense is this state vs regional average)
    df['state_vs_regional_ratio'] = np.where(
        df['regional_precip_mean'] > 0,
        df['precipitation'] / df['regional_precip_mean'],
        0
    )
    
    # Flag if THIS state had floods in past 3 days (backward-looking, no leakage)
    # Shift flood info by 1 day to ensure we don't use today's target
    df['recent_flood_history'] = df.groupby('state')['is_flood'].shift(1).rolling(3, min_periods=1).max()
    df['recent_flood_history'] = df['recent_flood_history'].fillna(0)
    
    return df


def silverman_bandwidth(X):
    """
    Calculate optimal bandwidth for KDE using Silverman's rule of thumb.
    
    Parameters
    ----------
    X : np.ndarray
        Input data with shape (n_samples, n_features)
    
    Returns
    -------
    float
        Recommended bandwidth for KernelDensity
    """
    n, d = X.shape
    sigma = np.std(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    
    # Use minimum of std and IQR/1.34
    width = np.minimum(sigma, iqr/1.34)
    width = np.where(width == 0, sigma, width)  # Fallback if IQR is 0
    
    # Silverman's factor
    h = 0.9 * width * (n ** (-1/(d+4)))
    return np.mean(h)  # Return average for multi-d


def train_kde_anomaly_detector(X_train, X_val, y_val, bandwidth=None):
    """
    Train a KDE-based anomaly detector with optimal threshold selection.
    
    Finds the best threshold by maximizing F1 score on validation set.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (n_samples, n_features)
    X_val : np.ndarray
        Validation feature matrix (n_samples, n_features)
    y_val : np.ndarray
        Validation labels (0=normal, 1=anomaly/flood)
    bandwidth : float, optional
        KDE bandwidth. If None, uses Silverman's rule of thumb
    
    Returns
    -------
    tuple
        (kde, best_threshold, bandwidth)
        - kde: Fitted KernelDensity object
        - best_threshold: Optimal anomaly score threshold
        - bandwidth: Bandwidth used for the kernel
    """
    if bandwidth is None:
        bandwidth = silverman_bandwidth(X_train)
    
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(X_train)
    
    # Score validation set (negative log-likelihood = anomaly score)
    val_scores = -kde.score_samples(X_val)
    
    # Find optimal threshold using F1 (binary: 1 = positive class)
    thresholds = np.percentile(val_scores, np.linspace(80, 99, 50))
    best_f1 = 0
    best_thr = thresholds[0]
    
    for thr in thresholds:
        y_pred = (val_scores > thr).astype(int)
        f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    
    return kde, best_thr, bandwidth


def compute_metrics(y_true, y_pred, scores):
    """
    Compute comprehensive classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted binary labels
    scores : np.ndarray
        Anomaly scores or probabilities (for AUC/AP calculation)
    
    Returns
    -------
    dict
        Dictionary with keys: 'Precision', 'Recall', 'F1', 'AUC-ROC', 'AP'
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )
    
    return {
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'AUC-ROC': roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0,
        'AP': average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0,
    }


def apply_log_transformation(df, feature_list):
    """
    Apply log transformation to a list of features using their mean.
    
    Uses logxp(value) = log(mean + value) instead of log1p to leverage negative numbers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_list : list
        List of feature names to transform
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added log-transformed features (appended with '_log' suffix)
    """
    df = df.copy()
    
    for col in feature_list:
        if col in df.columns:
            # Calculate mean for this feature
            x_mean = df[col].mean()
            # Apply logxp transformation: log(mean + value)
            df[f'{col}_log'] = np.log(x_mean + df[col])
    
    return df


def add_cyclical_features(df):
    """
    Add cyclical time encoding for seasonality.
    
    Encodes day-of-year as sine and cosine to capture circularity without
    discontinuity at year boundaries.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'date' column (datetime)
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added features:
        - day_of_year: Day number (1-365)
        - day_sin: Sine-encoded day (captures seasonal cycle)
        - day_cos: Cosine-encoded day (captures seasonal cycle)
        - month: Month number (1-12)
    """
    df = df.copy()
    
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month'] = df['date'].dt.month
    
    return df


def prepare_features(df, feature_cols, scaler=None, fit=True):
    """
    Prepare feature matrix and optional scaling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of column names to use as features
    scaler : StandardScaler, optional
        Fitted scaler for transformation. If None and fit=True, creates new one.
    fit : bool, default=True
        If True and scaler is None, fits scaler on data.
        If False, assumes scaler is already fitted.
    
    Returns
    -------
    tuple
        (X_scaled, scaler)
        - X_scaled: Scaled feature matrix
        - scaler: StandardScaler object (either input or newly fitted)
    """
    X = df[feature_cols].values
    
    if scaler is None:
        scaler = StandardScaler()
        if fit:
            X_scaled = scaler.fit_transform(X)
        else:
            raise ValueError("scaler cannot be None if fit=False")
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


if __name__ == "__main__":
    print("Texas Flood Detection Utilities")
    print("Available functions:")
    print("  - add_temporal_features")
    print("  - add_regional_features")
    print("  - silverman_bandwidth")
    print("  - train_kde_anomaly_detector")
    print("  - compute_metrics")
    print("  - apply_log_transformation")
    print("  - add_cyclical_features")
    print("  - prepare_features")
