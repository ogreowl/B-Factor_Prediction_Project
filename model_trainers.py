# src/model_trainers.py

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ======================
# Linear Models Trainer
# ======================
class LinearTrainer:
    def __init__(self, model_type: str = 'linear', alpha: float = 1.0):
        """Initialize linear model (Linear, Ridge, or Lasso)."""
        if model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()

    def train(self, X_train, y_train):
        """Fit model to training data."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate model on test data, returning Pearson correlation and MSE."""
        y_pred = self.model.predict(X_test)
        pearson_corr, _ = pearsonr(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return {'pearson': pearson_corr, 'mse': mse}


# ======================
# RNN Trainer
# ======================
class RNNTrainer(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, num_layers=2, dropout=0.2, learning_rate=0.001, n_epochs=50):
        """Initialize RNN model."""
        super(RNNTrainer, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def forward(self, x):
        """Forward pass for RNN."""
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

    def train_model(self, X_train, y_train):
        """Train the RNN model."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add batch dimension
        y_train_tensor = torch.FloatTensor(y_train)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.n_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate the RNN model."""
        self.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
            y_test_tensor = torch.FloatTensor(y_test)
            predictions = self(X_test_tensor).numpy()
            pearson_corr, _ = pearsonr(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
        return {'pearson': pearson_corr, 'mse': mse}


# ======================
# LSTM Trainer
# ======================
class LSTMTrainer(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, num_layers=2, dropout=0.2, learning_rate=0.001, n_epochs=50):
        """Initialize LSTM model."""
        super(LSTMTrainer, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def forward(self, x):
        """Forward pass for LSTM."""
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

    def train_model(self, X_train, y_train):
        """Train the LSTM model."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
        y_train_tensor = torch.FloatTensor(y_train)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.n_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate the LSTM model."""
        self.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
            y_test_tensor = torch.FloatTensor(y_test)
            predictions = self(X_test_tensor).numpy()
            pearson_corr, _ = pearsonr(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
        return {'pearson': pearson_corr, 'mse': mse}


# ======================
# Transformer Trainer
# ======================
class TransformerTrainer(nn.Module):
    def __init__(self, input_dim=1024, d_model=256, num_heads=8, num_layers=2, d_ff=512, dropout=0.1, learning_rate=0.001, n_epochs=50):
        """Initialize Transformer model."""
        super(TransformerTrainer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.input_projection = nn.Linear(input_dim, d_model)

    def forward(self, x):
        """Forward pass for Transformer."""
        x = self.input_projection(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out.squeeze()

    def train_model(self, X_train, y_train):
        """Train the Transformer model."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
        y_train_tensor = torch.FloatTensor(y_train)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.n_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate the Transformer model."""
        self.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
            y_test_tensor = torch.FloatTensor(y_test)
            predictions = self(X_test_tensor).numpy()
            pearson_corr, _ = pearsonr(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
        return {'pearson': pearson_corr, 'mse': mse}
``
