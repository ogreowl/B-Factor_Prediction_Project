
import matplotlib.pyplot as plt
import seaborn as sns

# Set a consistent style for all plots
sns.set(style="whitegrid")

def plot_predictions(y_true, y_pred, title: str = "Predicted vs Actual B-Factors", xlabel: str = "Actual B-Factors", ylabel: str = "Predicted B-Factors"):
    """Scatter plot of predicted vs actual B-Factor values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor='k')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_loss_curves(train_losses, val_losses, title: str = "Training and Validation Loss"):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_model_comparison(metrics_dict, metric_name="Pearson Correlation", title="Model Comparison"):
    """Bar plot for comparing models based on a specific metric (e.g., Pearson correlation, MSE).
    
    Args:
        metrics_dict (dict): Dictionary where keys are model names and values are the metric scores.
        metric_name (str): The name of the metric (e.g., "Pearson Correlation", "MSE").
    """
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=metrics)
    plt.title(f"{title} ({metric_name})")
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.show()
