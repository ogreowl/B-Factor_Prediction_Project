# src/hyperparameter_tuning.py

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader, TensorDataset

# ======================
# Hyperparameter Tuning for Scikit-Learn Models
# ======================
def pearson_scorer(y_true, y_pred):
    """Custom scorer for Pearson correlation."""
    return pearsonr(y_true, y_pred)[0]

def tune_sklearn_model(model, X, y, param_grid, scoring='pearson', n_jobs=-1, cv=5):
    """Tunes hyperparameters for scikit-learn models using GridSearchCV.
    
    Args:
        model: The scikit-learn model (e.g., Ridge, Lasso, LinearRegression).
        X: Training data features.
        y: Training data labels.
        param_grid: Dictionary of parameters to tune.
        scoring: Metric to optimize ('pearson' for Pearson correlation or 'neg_mean_squared_error').
        n_jobs: Number of jobs to run in parallel.
        cv: Number of cross-validation folds.

    Returns:
        dict: Best parameters and best score found during tuning.
    """
    if scoring == 'pearson':
        scorer = make_scorer(pearson_scorer, greater_is_better=True)
    else:
        scorer = 'neg_mean_squared_error'
    
    grid_search = GridSearchCV(model, param_grid, scoring=scorer, n_jobs=n_jobs, cv=cv)
    grid_search.fit(X, y)
    
    return {'best_params': grid_search.best_params_, 'best_score': grid_search.best_score_}


# ======================
# Hyperparameter Tuning for PyTorch Models
# ======================
def evaluate_model(model, X_test, y_test):
    """Helper function to evaluate a PyTorch model with Pearson correlation and MSE."""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
        y_test_tensor = torch.FloatTensor(y_test)
        predictions = model(X_test_tensor).numpy()
        pearson_corr, _ = pearsonr(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
    return {'pearson': pearson_corr, 'mse': mse}

def tune_torch_model(model_class, X_train, y_train, param_grid, n_epochs=50, batch_size=32, device='cpu'):
    """Manual grid search for PyTorch models.
    
    Args:
        model_class: Class of the PyTorch model (e.g., RNNTrainer, LSTMTrainer, TransformerTrainer).
        X_train: Training data features.
        y_train: Training data labels.
        param_grid: Dictionary of parameter ranges to explore.
        n_epochs: Number of epochs for training.
        batch_size: Batch size for training.
        device: Device to run training on ('cpu' or 'cuda').

    Returns:
        dict: Best parameters and best score found during tuning.
    """
    best_params = None
    best_score = -np.inf
    best_model = None
    
    # Convert data to PyTorch tensors and DataLoader
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for config in param_grid:
        # Instantiate model with current hyperparameters
        model = model_class(**config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
        criterion = torch.nn.MSELoss()
        
        # Training loop
        model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        # Evaluate model
        metrics = evaluate_model(model, X_train, y_train)
        pearson_corr = metrics['pearson']
        
        # Check if this is the best model so far
        if pearson_corr > best_score:
            best_score = pearson_corr
            best_params = config
            best_model = model
    
    return {'best_params': best_params, 'best_score': best_score, 'best_model': best_model}
