import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import time
import matplotlib.pyplot as plt
import random

# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For CUDA GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define transformer components
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)

        # Attention mechanism
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, ff_dim=128, dropout=0.1):
        super().__init__()

        # Layer normalisation and attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads=heads, dropout=dropout)

        # Layer normalisation and feed-forward network
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Apply attention with residual connection
        x = x + self.attn(self.norm1(x))

        # Apply feed-forward with residual connection
        x = x + self.ff(self.norm2(x))

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is [batch_size, seq_len, embedding_dim]
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=256, n_layers=14, heads=2,
                 dropout=0.2, ff_dim=256):
        super(TransformerModel, self).__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, heads=heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Global attention pooling
        self.global_attn = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )

        # Final output layers
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

        # Apply weight initialisation
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'weight' in name and len(p.shape) >= 2:
                nn.init.kaiming_normal_(p, nonlinearity='relu')

    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Project input to embedding dimension
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Apply layer normalisation
        x = self.norm(x)

        # Apply global attention pooling
        attn_weights = self.global_attn(x)
        x = torch.sum(x * attn_weights, dim=1)

        # Project to output dimension
        x = self.output_proj(x)

        return x

# Training and evaluation functions
def train_epoch(model, train_loader, criterion, optimiser, device, clip_value=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Forward pass
        optimiser.zero_grad()
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimiser.step()

        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def validate(model, val_loader, criterion, device):
    """Validate model performance"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # Forward pass
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)

            total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss

def evaluate_transformer_model(model, test_loader, device, target_proteins, scaler_Y):
    """Evaluate transformer model on test set"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # Forward pass
            Y_pred = model(X_batch)

            # Store predictions and targets
            all_predictions.append(Y_pred.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())

    # Concatenate batches
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # Inverse transform to original scale
    all_predictions = scaler_Y.inverse_transform(all_predictions)
    all_targets = scaler_Y.inverse_transform(all_targets)

    # Calculate metrics per feature
    feature_metrics = []

    for i, feature_name in enumerate(target_proteins):
        mse = mean_squared_error(all_targets[:, i], all_predictions[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
        r2 = r2_score(all_targets[:, i], all_predictions[:, i])

        # Calculate Pearson correlation
        pearson_corr, p_value = stats.pearsonr(all_targets[:, i], all_predictions[:, i])

        feature_metrics.append({
            'feature': feature_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson': pearson_corr,
            'p_value': p_value
        })

    # Overall metrics
    overall_mse = mean_squared_error(all_targets, all_predictions)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = mean_absolute_error(all_targets, all_predictions)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(feature_metrics)

    return {
        'feature_metrics': metrics_df,
        'overall_mse': overall_mse,
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'predictions': all_predictions,
        'targets': all_targets
    }

def train_and_evaluate_transformer_on_folds(d_model, n_layers, heads, fold_data, target_proteins, scaler_Y, device=None):
    """
    Train and evaluate transformer model

    Parameters:
    - fold_data: List of dictionaries with fold data
    - target_proteins: List of protein names (targets)
    - scaler_Y: Scaler used for target data
    - device: PyTorch device (defaults to CUDA if available)

    Returns:
    - Dictionary with transformer model results for each fold
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Store results for each fold
    transformer_fold_results = []

    # Get input and output dimensions from the first fold
    first_fold = fold_data[0]
    input_dim = first_fold['X_train_oversampled'].shape[1]
    output_dim = first_fold['Y_train_oversampled'].shape[1]

    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    # Process each fold
    for fold_info in fold_data:
        fold_num = fold_info['fold']
        print(f"\nProcessing Fold {fold_num}:")

        # Set seeds for reproducibility
        torch.manual_seed(42 + fold_num)
        np.random.seed(42 + fold_num)

        # Extract data for this fold
        X_train = fold_info['X_train_oversampled']
        Y_train = fold_info['Y_train_oversampled']
        X_val = fold_info['X_val_fold']
        Y_val = fold_info['Y_val_fold']
        X_test = fold_info['X_test_scaled']
        Y_test = fold_info['Y_test_scaled']

        # Create transformer model
        model = TransformerModel(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            n_layers=n_layers,
            heads=heads,
            dropout=0.2,
            ff_dim=256
        ).to(device)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        Y_train_tensor = torch.FloatTensor(Y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        Y_val_tensor = torch.FloatTensor(Y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        Y_test_tensor = torch.FloatTensor(Y_test)

        # Create datasets and dataloaders
        batch_size = 16
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Setup training components
        criterion = nn.MSELoss()
        optimiser = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Training loop settings
        num_epochs = 150
        early_stopping_patience = 10
        best_val_loss = float('inf')
        no_improvement_count = 0
        best_model_state = None

        # Lists to track losses
        train_losses = []
        val_losses = []

        # Training loop
        print("Starting training...")
        for epoch in range(num_epochs):
            # Train and validate
            train_loss = train_epoch(model, train_loader, criterion, optimiser, device)
            val_loss = validate(model, val_loader, criterion, device)

            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Update learning rate based on validation loss
            scheduler.step(val_loss)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model for evaluation
        model.load_state_dict(best_model_state)

        # Evaluate on test set
        test_results = evaluate_transformer_model(
            model, test_loader, device, target_proteins, scaler_Y
        )

        # Add fold info and training history
        test_results['fold'] = fold_num
        test_results['train_losses'] = train_losses
        test_results['val_losses'] = val_losses
        test_results['final_epoch'] = len(train_losses)
        test_results['best_val_loss'] = best_val_loss

        # Store results for this fold
        transformer_fold_results.append(test_results)

        print(f"Fold {fold_num} completed. Best validation loss: {best_val_loss:.6f}")
        print(f"Overall test MSE: {test_results['overall_mse']:.6f}")

    return transformer_fold_results

def aggregate_fold_results(fold_results):
    """
    Aggregate metrics across CV folds

    Parameters:
    - fold_results: List of dictionaries with results from each fold

    Returns:
    - Dictionary with aggregated metrics
    """
    # Aggregate overall metrics
    overall_mse = np.mean([result['overall_mse'] for result in fold_results])
    overall_rmse = np.mean([result['overall_rmse'] for result in fold_results])
    overall_mae = np.mean([result['overall_mae'] for result in fold_results])

    # Get features from first fold
    first_fold_features = fold_results[0]['feature_metrics']['feature'].tolist()

    # Aggregate feature-wise metrics
    aggregated_features = []

    for feature_name in first_fold_features:
        # Get metrics for this feature across all folds
        feature_metrics = {}

        for metric in ['mse', 'rmse', 'mae', 'r2', 'pearson']:
            # Extract values for this metric and feature across all folds
            values = []
            for fold_result in fold_results:
                fold_df = fold_result['feature_metrics']
                value = fold_df.loc[fold_df['feature'] == feature_name, metric].values[0]
                values.append(value)

            # Average values across folds
            feature_metrics[metric] = np.mean(values)

        # Add to aggregated features
        aggregated_features.append({
            'feature': feature_name,
            **feature_metrics
        })

    return {
        'overall_mse': overall_mse,
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'feature_metrics': pd.DataFrame(aggregated_features)
    }

def plot_training_curves(fold_results):
    """Plot training and validation loss curves for all folds"""
    plt.figure(figsize=(12, 5))

    # Plot training losses
    plt.subplot(1, 2, 1)
    for i, result in enumerate(fold_results):
        plt.plot(result['train_losses'], label=f'Fold {result["fold"]}')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot validation losses
    plt.subplot(1, 2, 2)
    for i, result in enumerate(fold_results):
        plt.plot(result['val_losses'], label=f'Fold {result["fold"]}')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def compare_with_baselines(transformer_results, baseline_results):
    """
    Compare transformer results with baseline models

    Parameters:
    - transformer_results: Dictionary with aggregated transformer results
    - baseline_results: Dictionary with aggregated results for each baseline model type

    Returns:
    - DataFrame with comparison in a clean, pivoted format
    """
    # Initialise comparison data
    comparison_data = []

    # Add overall metrics
    for metric in ['overall_mse', 'overall_rmse', 'overall_mae']:
        # Add transformer result
        comparison_data.append({
            'metric': metric,
            'model': 'transformer',
            'value': transformer_results[metric]
        })

        # Add baseline model results
        for model_name, results in baseline_results.items():
            comparison_data.append({
                'metric': metric,
                'model': model_name,
                'value': results[metric]
            })

    # Add feature-wise metrics
    features = transformer_results['feature_metrics']['feature'].tolist()

    for feature in features:
        for metric in ['mse', 'rmse', 'mae', 'r2', 'pearson']:
            metric_name = f"{feature}_{metric}"

            # Add transformer result
            tr_value = transformer_results['feature_metrics'].loc[
                transformer_results['feature_metrics']['feature'] == feature, metric
            ].values[0]

            comparison_data.append({
                'metric': metric_name,
                'model': 'transformer',
                'value': tr_value
            })

            # Add baseline model results
            for model_name, results in baseline_results.items():
                baseline_value = results['feature_metrics'].loc[
                    results['feature_metrics']['feature'] == feature, metric
                ].values[0]

                comparison_data.append({
                    'metric': metric_name,
                    'model': model_name,
                    'value': baseline_value
                })

    # Convert to DataFrame
    df = pd.DataFrame(comparison_data)

    # Pivot the data for easier comparison
    pivoted_df = df.pivot(index='metric', columns='model', values='value')

    # Calculate percentage improvements
    for baseline in baseline_results.keys():
        # For error metrics (lower is better)
        for metric in ['overall_mse', 'overall_rmse', 'overall_mae']:
            if metric in pivoted_df.index:
                improvement = (pivoted_df.loc[metric, baseline] - pivoted_df.loc[metric, 'transformer']) / pivoted_df.loc[metric, baseline] * 100
                pivoted_df.loc[f"{metric}_improvement_vs_{baseline}", 'transformer'] = improvement

        # Add feature-wise improvements
        for feature in features:
            for error_metric in ['mse', 'rmse', 'mae']:
                metric = f"{feature}_{error_metric}"
                if metric in pivoted_df.index:
                    improvement = (pivoted_df.loc[metric, baseline] - pivoted_df.loc[metric, 'transformer']) / pivoted_df.loc[metric, baseline] * 100
                    pivoted_df.loc[f"{metric}_improvement_vs_{baseline}", 'transformer'] = improvement

            # For R² and Pearson (higher is better)
            for score_metric in ['r2', 'pearson']:
                metric = f"{feature}_{score_metric}"
                if metric in pivoted_df.index:
                    improvement = (pivoted_df.loc[metric, 'transformer'] - pivoted_df.loc[metric, baseline]) / pivoted_df.loc[metric, baseline] * 100
                    pivoted_df.loc[f"{metric}_improvement_vs_{baseline}", 'transformer'] = improvement

    return pivoted_df

