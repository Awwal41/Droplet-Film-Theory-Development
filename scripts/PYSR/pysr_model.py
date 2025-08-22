# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from pysr import PySRRegressor
import matplotlib as mpl
import os

# Set matplotlib configuration for better compatibility
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Try to set LaTeX if available, otherwise use default
try:
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern']
except:
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'sans-serif'

# Load data with proper path handling
data_path = os.path.join(os.path.dirname(__file__), '..', 'processed_well_data.csv')
if not os.path.exists(data_path):
    # Try alternative path
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'processed_well_data.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find processed_well_data.csv. Tried paths: {data_path}")

print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

# Define features and targets
features = ['Dia', 'Dev_deg', 'Area_m2', 'z', 'GasDens', 'LiquidDens', 'g_m_s2', 'P_T', 'friction_factor', 'critical_film_thickness']
output = 'Qcr'
gasflow = 'Gasflowrate'
status_col = 'Test status'

# Clean up column names in dataframe
df.rename(columns={
    'Dev(deg)': 'Dev_deg',
    'Area (m2)': 'Area_m2',
    'g (m/s2)': 'g_m_s2',
    'P/T': 'P_T'
}, inplace=True)

# Map Test status to numerical classes
loading_class_map = {'Unloaded': -1, 'Near L.U': 0, 'Loaded': 1, 'Questionable': 1}
df['loading_class'] = df[status_col].map(loading_class_map)

# Split data
X = df[features]
y = df[output]
gsflow = df[gasflow]
loading_class = df['loading_class']
X_train, X_test, y_train, y_test, gsflow_train, gsflow_test, loading_train, loading_test = train_test_split(
    X, y, gsflow, loading_class, test_size=0.2, random_state=42, stratify=loading_class
)

# Scale features and target
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
X_scaled = scaler_X.transform(X)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
y_scaled = scaler_y.transform(y.values.reshape(-1, 1)).flatten()

# Create DataFrames for PySR
train_df = pd.DataFrame(X_train_scaled, columns=features)
train_df[output] = y_train_scaled
test_df = pd.DataFrame(X_test_scaled, columns=features)
test_df[output] = y_test_scaled
full_df = pd.DataFrame(X_scaled, columns=features)
full_df[output] = y_scaled

# Accuracy calculation
def calculate_accuracy(y_pred, gsflow, loading_actual, interval=0.01):
    y_pred = np.array(y_pred)
    gsflow = np.array(gsflow)
    loading_pred = np.where(y_pred > gsflow + interval, 1,
                           np.where(y_pred < gsflow - interval, -1, 0))
    return accuracy_score(loading_actual, loading_pred), confusion_matrix(loading_actual, loading_pred, labels=[-1, 0, 1])

# Cross-validation with PySR
def evaluate_pysr(train_df, loading_class, cv_splits=5, maxsize=15, niterations=100):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    acc_scores = []

    for train_idx, val_idx in cv.split(train_df[features], loading_class):
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        gsflow_val = gsflow_train.iloc[val_idx] if isinstance(gsflow_train, pd.Series) else gsflow_train[val_idx]
        loading_val = loading_class.iloc[val_idx] if isinstance(loading_class, pd.Series) else loading_class[val_idx]

        # Initialize PySR model
        model = PySRRegressor(
            niterations=niterations,
            maxsize=maxsize,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "exp", "log"],
            model_selection="best",
            elementwise_loss="loss(x, y) = (x - y)^2",
            annealing=True,
            random_state=42
        )

        # Fit model
        model.fit(train_fold[features], train_fold[output])

        # Predict on validation set
        y_val_pred_scaled = model.predict(val_fold[features])
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()

        acc, _ = calculate_accuracy(y_val_pred, gsflow_val, loading_val)
        acc_scores.append(acc)

    return np.mean(acc_scores)

# Plot confusion matrix
def plot_confusion_matrix(cm, title, filename=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Unloaded', 'Near L.U', 'Loaded'],
                yticklabels=['Unloaded', 'Near L.U', 'Loaded'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
    
    plt.show()
    plt.close()

# Evaluate with cross-validation
try:
    cv_score = evaluate_pysr(train_df, loading_train, maxsize=15, niterations=100)
    print(f"Cross-validation accuracy: {cv_score*100:.2f}%")
except Exception as e:
    print(f"Cross-validation failed: {e}")
    cv_score = 0.0

# Train final model
model = PySRRegressor(
    niterations=100,
    maxsize=15,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "exp", "log"],
    model_selection="best",
    annealing=True,
    random_state=42
)
model.fit(train_df[features], train_df[output])

# Predict on train and test sets
y_train_pred_scaled = model.predict(train_df[features])
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
y_test_pred_scaled = model.predict(test_df[features])
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

# Calculate metrics
train_acc, train_cm = calculate_accuracy(y_train_pred, gsflow_train, loading_train)
test_acc, test_cm = calculate_accuracy(y_test_pred, gsflow_test, loading_test)

print("\n=== Final Model Performance ===")
print("Mathematical Expression:")
print(model.equations_)
print(f"Training Set Classification Accuracy: {train_acc*100:.2f}%")
print(f"Test Set Classification Accuracy: {test_acc*100:.2f}%")
print("Confusion Matrix (Train):\n", train_cm)
print("Confusion Matrix (Test):\n", test_cm)

# Plot confusion matrices
plot_confusion_matrix(train_cm, "Confusion Matrix (Training Set)", "train_confusion_matrix.pdf")
plot_confusion_matrix(test_cm, "Confusion Matrix (Test Set)", "test_confusion_matrix.pdf")

# Create additional diagnostic plots
def create_diagnostic_plots(y_true, y_pred, title_prefix):
    """Create residual and prediction plots for model diagnostics"""
    
    # Residual plot
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax1.set_xlabel('Predicted Values', fontweight='bold')
    ax1.set_ylabel('Residuals', fontweight='bold')
    ax1.set_title(f'{title_prefix} - Residuals vs Predicted', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Residuals histogram
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuals', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title(f'{title_prefix} - Residuals Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{title_prefix.lower().replace(' ', '_')}_diagnostics.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# Create diagnostic plots for training and test sets
create_diagnostic_plots(y_train, y_train_pred, "Training Set")
create_diagnostic_plots(y_test, y_test_pred, "Test Set")

# Scatter plot
color_map = {
    'Loaded': '#FF6363',  # Light red
    'Unloaded': '#3674B5',  # Light purple
    'Questionable': '#D3D3D3',  # Light orange
    'Near L.U': '#FFCC99'  # Light blue
}
y_pred_scaled = model.predict(full_df[features])
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
colors = df[status_col].map(color_map).fillna('gray')

plt.figure(figsize=(10, 8))
plt.scatter(gsflow, y_pred, c=colors, alpha=1, s=100, edgecolors="gray", linewidth=0.5)
plt.plot([0, 350000], [0, 350000], '--', color='#FF6666', linewidth=1.5)
plt.title("PySR Model"  , fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Well measured flow rate (m³/day)", fontsize=14, fontweight='bold')
plt.ylabel("Critical rate (m³/day)", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(0, 350000)
plt.ylim(0, 350000)
legend_patches = [mpatches.Patch(color=color, label=status) for status, color in color_map.items()]
plt.legend(handles=legend_patches, title='Actual label', fontsize=12, title_fontsize=14, loc='upper left')
plt.tight_layout()

# Save and display the plot
plt.savefig("pysr_scatter.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# Print final summary
print("\n" + "="*60)
print("PYSR MODEL TRAINING COMPLETE")
print("="*60)
print(f"Cross-validation accuracy: {cv_score*100:.2f}%")
print(f"Training accuracy: {train_acc*100:.2f}%")
print(f"Test accuracy: {test_acc*100:.2f}%")
print(f"Best equation: {model.equations_[0] if hasattr(model, 'equations_') and len(model.equations_) > 0 else 'No equation found'}")
print("\nPlots saved:")
print("- train_confusion_matrix.pdf")
print("- test_confusion_matrix.pdf") 
print("- training_set_diagnostics.pdf")
print("- test_set_diagnostics.pdf")
print("- pysr_scatter.pdf")
print("="*60)