# ==== Core Python & Data Libraries ====
import os
import time
import warnings
import numpy as np
import pandas as pd
import random

# ==== Visualisation ====
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Sklearn: Preprocessing, Decomposition, Metrics, Models ====
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

# ==== Parallel Processing ====
from joblib import Parallel, delayed

# ==== Statistical Analysis ====
from scipy import stats

# ==== PyTorch ====
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings('ignore')

# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For CUDA GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#############################################################################   
                              # Data loaders
#############################################################################  

# Function to load and preprocess datasets
def load_dataset(file_path, set_index=True):
    try:
        df = pd.read_csv(file_path)
        # Set the first column as index (cell IDs)
        if set_index and 'Unnamed: 0' in df.columns:
            df = df.set_index('Unnamed: 0')
        elif set_index and not df.columns[0].startswith('Unnamed'):
            df = df.set_index(df.columns[0])
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Get common samples across datasets
def get_common_samples(datasets_dict):
    all_indices = []
    for name, df in datasets_dict.items():
        if df is not None:
            all_indices.append(set(df.index))

    common_indices = set.intersection(*all_indices)
    print(f"Number of common samples across all datasets: {len(common_indices)}")
    return sorted(list(common_indices))

## Feature selection and pre-processing

# Function to select features based on variance with adjustable threshold
def select_high_variance_features(df, threshold_percentile=90):
    # Calculate variance for each feature
    variances = df.var()

    # Select features with variance above the percentile threshold
    threshold = np.percentile(variances, threshold_percentile)
    high_var_features = variances[variances > threshold].index

    print(f"Selected {len(high_var_features)} out of {len(variances)} features based on variance percentile {threshold_percentile}")
    return df[high_var_features]

# Function to select features using PCA
def select_pca_features(df, n_components=100):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Create a DataFrame with PCA components
    pca_df = pd.DataFrame(
        X_pca,
        index=df.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )

    # Calculate explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA with {n_components} components explains {explained_variance:.2f}% of variance")

    return pca_df, pca, scaler

# Function to find correlations between proteomics and transcriptomics
def find_gene_correlations(proteomics_df, transcriptomics_df):
    correlations = []

    # Get all protein features
    protein_features = proteomics_df.columns

    # For each protein feature, find the most correlated transcriptomics features
    for protein in protein_features:
        protein_gene = protein.split('_')[0] if '_' in protein else protein

        # Find all transcriptomics features for the same gene
        trans_features = [col for col in transcriptomics_df.columns
                         if (col.split('_')[0] if '_' in col else col) == protein_gene]

        if trans_features:
            # Calculate correlation with each matching transcriptomics feature
            for trans_feature in trans_features:
                corr = proteomics_df[protein].corr(transcriptomics_df[trans_feature])
                if not np.isnan(corr):
                    correlations.append((protein, trans_feature, corr))

    # Create DataFrame with correlations
    corr_df = pd.DataFrame(correlations, columns=['protein', 'transcript', 'correlation'])
    return corr_df

def create_oversampled_data(X, y, oversampling_factor=2):
    """
    Create oversampled dataset based on target value distributions
    """
    # Calculate response mean across all target proteins
    response_mean = np.mean(y, axis=1)

    # Create bins (5 equal-width bins)
    bins = pd.qcut(response_mean, 5, labels=False, duplicates='drop')
    bin_counts = np.bincount(bins)

    # Calculate weights (inverse frequency)
    weights = 1.0 / bin_counts[bins]

    # Normalise weights
    weights /= weights.sum()

    # Determine desired total samples
    total_samples = len(X) * oversampling_factor

    # Collect oversampled indices
    oversampled_indices = []

    # Ensure each bin is represented proportionally
    for bin_label in range(len(bin_counts)):
        bin_mask = (bins == bin_label)
        bin_indices = np.where(bin_mask)[0]

        # Calculate number of samples for this bin
        bin_samples = int(np.ceil(len(bin_indices) * oversampling_factor))

        # Sample with replacement from this bin
        bin_oversampled_indices = np.random.choice(
            bin_indices,
            size=bin_samples,
            replace=True
        )
        oversampled_indices.extend(bin_oversampled_indices)

    # Select oversampled data
    X_oversampled = X[oversampled_indices]
    y_oversampled = y[oversampled_indices]

    return X_oversampled, y_oversampled

def diagnose_oversampling(X_original, Y_original, X_oversampled, Y_oversampled):
    """
    Diagnose the effects of oversampling
    """
    # Calculate response mean for original and oversampled data
    response_mean_original = np.mean(Y_original, axis=1)
    response_mean_oversampled = np.mean(Y_oversampled, axis=1)

    # Create bins for original and oversampled data
    bins_original = pd.qcut(response_mean_original, 5, labels=False, duplicates='drop')
    bins_oversampled = pd.qcut(response_mean_oversampled, 5, labels=False, duplicates='drop')

    # Count samples in each bin
    bin_counts_original = np.bincount(bins_original)
    bin_counts_oversampled = np.bincount(bins_oversampled)

    print("\nOversampling Diagnosis:")
    print("Original Data Bin Counts:", bin_counts_original)
    print("Oversampled Data Bin Counts:", bin_counts_oversampled)

    # Visualise distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(response_mean_original, bins=20, edgecolor='black')
    plt.title('Original Response Mean Distribution')
    plt.xlabel('Mean Response')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(response_mean_oversampled, bins=20, edgecolor='black')
    plt.title('Oversampled Response Mean Distribution')
    plt.xlabel('Mean Response')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Additional diagnostics
    print("\nOriginal Data:")
    print(f"Number of samples: {len(X_original)}")
    print(f"Feature dimensions: {X_original.shape[1]}")

    print("\nOversampled Data:")
    print(f"Number of samples: {len(X_oversampled)}")
    print(f"Feature dimensions: {X_oversampled.shape[1]}")

    # Statistical comparison of original and oversampled data
    print("\nStatistical Comparison:")
    print("Original Data Mean:", np.mean(response_mean_original))
    print("Original Data Std:", np.std(response_mean_original))
    print("Oversampled Data Mean:", np.mean(response_mean_oversampled))
    print("Oversampled Data Std:", np.std(response_mean_oversampled))


#############################################################################   
                              # Evaluation
#############################################################################  


def evaluate_model(Y_pred, Y_true, target_proteins, scaler_Y, model_name):
    """
    Evaluate model performance - optimised version
    """
    # Inverse transform to original scale
    Y_pred_original = scaler_Y.inverse_transform(Y_pred)
    Y_true_original = scaler_Y.inverse_transform(Y_true)

    # Calculate overall metrics first
    overall_mse = mean_squared_error(Y_true_original, Y_pred_original)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = mean_absolute_error(Y_true_original, Y_pred_original)

    # Define function to compute metrics for a single feature
    def compute_feature_metrics(i, feature_name):
        true_vals = Y_true_original[:, i]
        pred_vals = Y_pred_original[:, i]

        mse = mean_squared_error(true_vals, pred_vals)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_vals, pred_vals)
        r2 = r2_score(true_vals, pred_vals)

        # Calculate Pearson correlation
        pearson_corr, p_value = stats.pearsonr(true_vals, pred_vals)

        return {
            'feature': feature_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson': pearson_corr,
            'p_value': p_value
        }

    feature_metrics = [compute_feature_metrics(i, feature_name)
                      for i, feature_name in enumerate(target_proteins)]

    # Convert to DataFrame
    metrics_df = pd.DataFrame(feature_metrics)

    return {
        'model_name': model_name,
        'feature_metrics': metrics_df,
        'overall_mse': overall_mse,
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'predictions': Y_pred_original,
        'targets': Y_true_original
    }


def aggregate_results(model_results):
    """
    Aggregate metrics across CV folds - optimised version
    """
    # Use numpy for faster aggregation
    overall_mse = np.mean([result['overall_mse'] for result in model_results])
    overall_rmse = np.mean([result['overall_rmse'] for result in model_results])
    overall_mae = np.mean([result['overall_mae'] for result in model_results])

    # Get features and create empty dataframe once
    first_fold_features = model_results[0]['feature_metrics']['feature'].tolist()
    metrics_columns = ['mse', 'rmse', 'mae', 'r2', 'pearson']

    # Initialise results array
    aggregated_data = {
        'feature': first_fold_features
    }

    # For each metric, pre-allocate arrays and compute means
    for metric in metrics_columns:
        # Create a 2D array: rows = features, cols = folds
        metric_values = np.zeros((len(first_fold_features), len(model_results)))

        for fold_idx, fold_result in enumerate(model_results):
            fold_df = fold_result['feature_metrics']
            for feat_idx, feature_name in enumerate(first_fold_features):
                metric_values[feat_idx, fold_idx] = fold_df.loc[
                    fold_df['feature'] == feature_name, metric].values[0]

        # Compute mean across folds for each feature
        aggregated_data[metric] = np.mean(metric_values, axis=1)

    # Create DataFrame all at once
    aggregated_features_df = pd.DataFrame(aggregated_data)

    return {
        'overall_mse': overall_mse,
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'feature_metrics': aggregated_features_df
    }

# Aggregate results across folds
def get_aggregated_results(results):
    aggregated_results = {}
    for model_name, model_results in results.items():
        aggregated_results[model_name] = aggregate_results(model_results)
    return aggregated_results


def compare_all_models(baseline_results, transformer_results):
    """
    Compare all model types: baseline, transformer

    Parameters:
    - baseline_results: Dictionary with aggregated baseline results
    - transformer_results: Dictionary with aggregated transformer results

    Returns:
    - DataFrame with comparison metrics
    """
    # Initialise data for comparison
    comparison_data = []

    # Add models to a single dictionary
    all_models = {
        **baseline_results,
        'transformer': transformer_results
    }

    # Add overall metrics
    for metric in ['overall_mse', 'overall_rmse', 'overall_mae']:
        for model_name, results in all_models.items():
            comparison_data.append({
                'metric': metric,
                'model': model_name,
                'value': results[metric]
            })

    # Get all features
    features = transformer_results['feature_metrics']['feature'].tolist()

    # Add feature-wise metrics
    for feature in features:
        for metric in ['mse', 'rmse', 'mae', 'r2', 'pearson']:
            metric_name = f"{feature}_{metric}"

            for model_name, results in all_models.items():
                model_df = results['feature_metrics']
                value = model_df.loc[model_df['feature'] == feature, metric].values[0]

                comparison_data.append({
                    'metric': metric_name,
                    'model': model_name,
                    'value': value
                })

    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Pivot
    pivoted_df = comparison_df.pivot(index='metric', columns='model', values='value')

    # Calculate improvement percentages for error metrics
    # Using linear regression as reference
    if 'linear_regression' in pivoted_df.columns:
        for model in ['transformer', 'vae', 'ridge_regression']:
            if model in pivoted_df.columns:
                for metric in ['overall_mse', 'overall_rmse', 'overall_mae']:
                    # Calculate improvement (negative means worse performance)
                    improvement = ((pivoted_df.loc[metric, 'linear_regression'] -
                                    pivoted_df.loc[metric, model]) /
                                   pivoted_df.loc[metric, 'linear_regression'] * 100)

                    pivoted_df.loc[f"{metric}_improvement_vs_linear", model] = improvement

                # Feature-wise improvement for error metrics
                for feature in features:
                    for error_metric in ['mse', 'rmse', 'mae']:
                        metric = f"{feature}_{error_metric}"

                        improvement = ((pivoted_df.loc[metric, 'linear_regression'] -
                                        pivoted_df.loc[metric, model]) /
                                       pivoted_df.loc[metric, 'linear_regression'] * 100)

                        pivoted_df.loc[f"{metric}_improvement_vs_linear", model] = improvement

                    # For metrics where higher is better (r2, pearson)
                    for score_metric in ['r2', 'pearson']:
                        metric = f"{feature}_{score_metric}"

                        improvement = ((pivoted_df.loc[metric, model] -
                                        pivoted_df.loc[metric, 'linear_regression']) /
                                       pivoted_df.loc[metric, 'linear_regression'] * 100)

                        pivoted_df.loc[f"{metric}_improvement_vs_linear", model] = improvement

    return pivoted_df

def visualise_model_comparison(comparison_df):
    """Create visualisation of model comparison metrics with Set2 colors"""
    # Extract overall metrics
    overall_metrics = ['overall_mse', 'overall_rmse', 'overall_mae']
    overall_df = comparison_df.loc[overall_metrics]

    # Define Set2 palette mapped to model names
    set2_palette = sns.color_palette("Set2", n_colors=overall_df.columns.shape[0])
    color_dict = dict(zip(overall_df.columns, set2_palette))

    # Plot
    ax = overall_df.plot(kind='bar', figsize=(10, 6), color=[color_dict[col] for col in overall_df.columns])

    plt.title('Comparison of Overall Metrics Across Models')
    plt.ylabel('Metric Value')
    plt.xlabel('Metric Type')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()

#############################################################################   
                              # Models
#############################################################################  

def process_fold(fold_info, target_proteins, scaler_Y):
    """
    Process a single fold - can be run in parallel
    """
    fold_num = fold_info['fold']
    X_train = fold_info['X_train_oversampled']
    Y_train = fold_info['Y_train_oversampled']
    X_val = fold_info['X_val_fold']
    Y_val = fold_info['Y_val_fold']
    X_test = fold_info['X_test_scaled']
    Y_test = fold_info['Y_test_scaled']

    fold_results = {}

    # Linear Regression
    lr_model = MultiOutputRegressor(LinearRegression())
    lr_model.fit(X_train, Y_train)
    Y_pred = lr_model.predict(X_test)
    lr_results = evaluate_model(Y_pred, Y_test, target_proteins, scaler_Y, f"Linear Regression - Fold {fold_num}")
    fold_results['linear_regression'] = lr_results

    # Ridge Regression with optimised hyperparameter tuning
    best_ridge_model = None
    best_score = float('inf')

    # Pre-compute validation predictions for all alphas at once
    alphas = [0.1, 1.0, 10.0]
    ridge_models = [MultiOutputRegressor(Ridge(alpha=alpha)) for alpha in alphas]

    # Fit all models
    for model in ridge_models:
        model.fit(X_train, Y_train)

    # Evaluate on validation set
    val_predictions = [model.predict(X_val) for model in ridge_models]
    val_scores = [mean_squared_error(Y_val, pred) for pred in val_predictions]

    # Find best model
    best_index = np.argmin(val_scores)
    best_ridge_model = ridge_models[best_index]

    # Predict on test set using best model
    Y_pred = best_ridge_model.predict(X_test)
    ridge_results = evaluate_model(Y_pred, Y_test, target_proteins, scaler_Y, f"Ridge Regression - Fold {fold_num}")
    fold_results['ridge_regression'] = ridge_results

    return fold_results

def run_baselines_on_folds(fold_data, target_proteins, scaler_Y, n_jobs=-1):
    """
    Run linear and ridge regression on each fold in parallel

    Parameters:
    - fold_data: List of fold information dictionaries
    - target_proteins: List of protein names
    - scaler_Y: Scaler used to transform target values
    - n_jobs: Number of jobs for parallel processing (-1 uses all cores)

    Returns:
    - Dictionary of results for each model type
    """
    # Process folds in parallel
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(process_fold)(fold_info, target_proteins, scaler_Y)
        for fold_info in fold_data
    )

    # Reorganise results by model type
    results = {
        'linear_regression': [res['linear_regression'] for res in fold_results],
        'ridge_regression': [res['ridge_regression'] for res in fold_results]
    }

    return results

# Visualisations

# Print aggregated results
def print_results(aggregated_results):
    print("\n===== AGGREGATED RESULTS =====")
    for model_name, agg_result in aggregated_results.items():
        print(f"\n{model_name.upper()} Cross-Validation Results:")
        print(f"Overall MSE: {agg_result['overall_mse']:.6f}")
        print(f"Overall RMSE: {agg_result['overall_rmse']:.6f}")
        print(f"Overall MAE: {agg_result['overall_mae']:.6f}")
        print("\nFeature-wise Metrics:")
        print(agg_result['feature_metrics'].sort_values(by='mae').head(20))

