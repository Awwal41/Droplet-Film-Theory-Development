Troubleshooting Guide
=====================

Droplet-Film Model Development Project
Common Issues and Solutions

Introduction
This guide addresses the most frequently encountered problems when using the DFT Development project. Each issue includes detailed diagnostic steps, multiple solution approaches, and prevention strategies.

The guide is organized by problem category, with solutions ranging from quick fixes to comprehensive debugging approaches.

Installation and Environment Issues

Issue 1: Python Import Errors
Problem: ModuleNotFoundError when importing project modules

Symptoms:
- "ModuleNotFoundError: No module named 'src'" or "No module named 'models'"
- "ImportError: cannot import name 'DFT'"
- "ModuleNotFoundError: No module named 'feyn'"

Root Causes:
- Virtual environment not activated
- Missing dependencies
- Incorrect Python path
- Package installation failures

Diagnostic Steps:
1. Check Python environment:

.. code-block:: bash

   python --version
   which python


2. Verify virtual environment activation:

.. code-block:: bash

   echo $VIRTUAL_ENV  # Linux/Mac
   echo %VIRTUAL_ENV%  # Windows


3. Test basic imports:

.. code-block:: bash

   python -c "import numpy; print('NumPy OK')"
   python -c "import pandas; print('Pandas OK')"


Solutions:

Solution 1: Activate Virtual Environment

.. code-block:: bash

   # Windows
   dft_env\Scripts\activate
   # macOS/Linux
   source dft_env/bin/activate

Solution 2: Install Missing Dependencies

.. code-block:: bash

   pip install numpy pandas scipy scikit-learn matplotlib seaborn feyn pysindy

Solution 3: Install Project in Development Mode

.. code-block:: bash

   cd /path/to/Droplet-Film-Model-Development
   pip install -e .

Solution 4: Fix Python Path

.. code-block:: python

   import sys
   sys.path.append('/path/to/Droplet-Film-Model-Development')

Prevention:
- Always use virtual environments
- Document exact package versions
- Use requirements.txt for reproducibility

Issue 2: Data Loading and Format Errors
Problem: Errors when loading or processing datasets

Symptoms:
- "FileNotFoundError: [Errno 2] No such file or directory"
- "KeyError: 'Dia'"
- "ValueError: could not convert string to float"
- "pandas.errors.EmptyDataError"

Root Causes:
- Incorrect file paths
- Missing required columns
- Data format inconsistencies
- File permission issues

Diagnostic Steps:
1. Verify file existence:

.. code-block:: python

   import os
   print(os.path.exists("your_data.csv"))
   print(os.path.abspath("your_data.csv"))


2. Check file format:

.. code-block:: python

   import pandas as pd
   data = pd.read_csv("your_data.csv", nrows=5)
   print(data.columns.tolist())
   print(data.dtypes)


3. Validate data content:

.. code-block:: python

   print(data.head())
   print(data.describe())


Solutions:

Solution 1: Fix File Paths

.. code-block:: python

   # Use absolute paths
   data_path = os.path.abspath("datasets/well_data.csv")
   # Check current working directory
   print(os.getcwd())

Solution 2: Validate Required Columns

.. code-block:: python

   required_cols = ['Dia', 'Dev(deg)', 'Area (m2)', 'z',
                    'GasDens', 'LiquidDens', 'g (m/s2)',
                    'P/T', 'friction_factor', 'critical_film_thickness']
   missing_cols = [col for col in required_cols if col not in data.columns]
   if missing_cols:
       print(f"Missing columns: {missing_cols}")
       for col in missing_cols:
           data[col] = 0.0

Solution 3: Handle Data Format Issues

.. code-block:: python

   # Clean data before processing
   data = data.dropna()
   data = data.replace([np.inf, -np.inf], np.nan).dropna()
   numeric_cols = ['Dia', 'Dev(deg)', 'Area (m2)', 'z',
                   'GasDens', 'LiquidDens', 'g (m/s2)',
                   'P/T', 'friction_factor', 'critical_film_thickness']
   for col in numeric_cols:
       data[col] = pd.to_numeric(data[col], errors='coerce')

Solution 4: Fix File Permissions

.. code-block:: python

   import stat
   file_stat = os.stat("your_data.csv")
   print(f"Readable: {bool(file_stat.st_mode & stat.S_IRUSR)}")
   print(f"Writable: {bool(file_stat.st_mode & stat.S_IWUSR)}")

Prevention:
- Use consistent file naming conventions
- Validate data before processing
- Implement data quality checks
- Use version control for datasets

Issue 3: Memory and Performance Issues
Problem: Out of memory or slow performance

Symptoms:
- "MemoryError: Unable to allocate array"
- "Killed: 9" (Linux/Mac)
- Extremely slow training times
- System becomes unresponsive

Root Causes:
- Dataset too large for available memory
- Inefficient data structures
- Large hyperparameter grids
- Memory leaks in optimization

Diagnostic Steps:
1. Check system memory:

.. code-block:: python

   import psutil
   print(f"Available memory: {psutil.virtual_memory().available / 1e9:.1f} GB")


2. Monitor memory usage:

.. code-block:: python

   import tracemalloc
   tracemalloc.start()
   # Your code here
   current, peak = tracemalloc.get_traced_memory()
   print(f"Current memory usage: {current / 1e6:.1f} MB")
   print(f"Peak memory usage: {peak / 1e6:.1f} MB")


3. Profile data size:

.. code-block:: python

   print(f"Dataset shape: {data.shape}")
   print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1e6:.1f} MB")


Solutions:

Solution 1: Reduce Dataset Size

.. code-block:: python

   data_sample = data.sample(n=1000, random_state=42)
   from sklearn.model_selection import train_test_split
   X_sample, _, y_sample, _ = train_test_split(
       X, y, train_size=1000, random_state=42, stratify=y
   )

Solution 2: Optimize Hyperparameter Grid

.. code-block:: python

   hparam_grid = {
       "dev_tol": [1e-3],
       "feature_tol": [1.0],
       "multiple_dev_policy": ["max"]
   }
   k_folds = 3

Solution 3: Process Data in Chunks

.. code-block:: python

   chunk_size = 1000
   results = []
   for i in range(0, len(data), chunk_size):
       chunk = data[i:i+chunk_size]
       chunk_result = process_chunk(chunk)
       results.append(chunk_result)

Solution 4: Use Memory-Efficient Data Types

.. code-block:: python

   data = data.astype({
       'Dia': 'float32',
       'Dev(deg)': 'float32',
       'Area (m2)': 'float32'
   })

Prevention:
- Start with small datasets during development
- Monitor memory usage regularly
- Use appropriate data types
- Implement data streaming for large files

Issue 4: Optimization Convergence Problems
Problem: Model training fails to converge

Symptoms:
- "RuntimeError: Optimization failed"
- "ConvergenceWarning: Optimization terminated early"
- Very poor model performance
- Unrealistic parameter values

Root Causes:
- Poor initial parameter values
- Inappropriate optimization bounds
- Insufficient iterations
- Numerical instability
- Poor data quality

Diagnostic Steps:
1. Check optimization status:

.. code-block:: python

   result = minimize(...)
   print(f"Success: {result.success}")
   print(f"Message: {result.message}")
   print(f"Function evaluations: {result.nfev}")


2. Analyze parameter values:

.. code-block:: python

   print(f"Parameter range: {params.min():.6f} to {params.max():.6f}")
   print(f"Parameter mean: {params.mean():.6f}")


3. Check data quality:

.. code-block:: python

   print(f"Data range: {X.min():.6f} to {X.max():.6f}")
   print(f"Data mean: {X.mean():.6f}")
   print(f"NaN values: {np.isnan(X).sum()}")


Solutions:

Solution 1: Adjust Optimization Parameters

.. code-block:: python

   result = minimize(
       self._loss, x0=x0, bounds=bounds, method="Powell",
       options={'maxiter': 10000, 'maxfun': 20000}
   )

Solution 2: Improve Initial Parameters

.. code-block:: python

   x0 = np.concatenate((
       [0.1, 0.1, 0.1, 0.1, 0.1],
       np.full(n_train, 0.5)
   ))
   p1_init = np.sqrt(np.mean(y) / np.mean(X[:, 0]))
   x0[0] = p1_init

Solution 3: Scale Input Data

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   model.fit(X_scaled, y)

Solution 4: Add Numerical Stability

.. code-block:: python

   epsilon = 1e-8
   z = np.maximum(z, epsilon)
   GasDens = np.maximum(GasDens, epsilon)

   def log_loss(log_params):
       params = np.exp(log_params)
       return self._loss(params)


Prevention:
- Always scale input data
- Use appropriate initial values
- Monitor optimization progress
- Validate data quality before training

Issue 5: QLattice Connection Problems
Problem: Cannot connect to Feyn QLattice service

Symptoms:
- "ConnectionError: Failed to connect to QLattice"
- "TimeoutError: Connection timed out"
- "AuthenticationError: Invalid credentials"

Root Causes:
- Network connectivity issues
- Invalid API credentials
- Service unavailability
- Firewall restrictions

Diagnostic Steps:
1. Test network connectivity:

.. code-block:: python

   import requests
   try:
   response = requests.get("https://api.feynlab.com", timeout=10)
   print(f"Connection status: {response.status_code}")
   except Exception as e:
   print(f"Connection failed: {e}")


2. Check Feyn configuration:

.. code-block:: python

   import feyn
   print(f"Feyn version: {feyn.__version__}")


3. Test QLattice connection:

.. code-block:: python

   try:
   ql = feyn.connect_qlattice()
   print("QLattice connection successful")
   except Exception as e:
   print(f"QLattice connection failed: {e}")


Solutions:

Solution 1: Check Network Settings

.. code-block:: python

   import socket
   try:
       socket.create_connection(("api.feynlab.com", 443), timeout=10)
       print("Network connection OK")
   except OSError:
       print("Network connection failed")
   import os
   print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY', 'Not set')}")
   print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', 'Not set')}")

Solution 2: Verify API Credentials

.. code-block:: python

   import feyn
   try:
       ql = feyn.connect_qlattice()
       print("Authentication successful")
   except Exception as e:
       print(f"Authentication failed: {e}")

Solution 3: Use Alternative Symbolic Regression

.. code-block:: python

   from pysindy import SINDy
   sindy_model = SINDy(
       optimizer='STLSQ',
       feature_library='polynomial',
       feature_names=builder.feature_names
   )
   sindy_model.fit(X_train, y_train)
   print(sindy_model.equations())

Solution 4: Implement Offline Mode

.. code-block:: python

   class OfflineQLatticeWrapper:
       def __init__(self, feature_tags, output_tag="Qcr"):
           self.feature_tags = feature_tags
           self.output_tag = output_tag
           self.model = None

       def fit(self, X, y):
           from sklearn.linear_model import LinearRegression
           self.model = LinearRegression()
           self.model.fit(X, y)

       def predict(self, X):
           return self.model.predict(X)

       def express(self):
           return "Linear regression (offline mode)"


Prevention:
- Test QLattice connection during installation
- Implement fallback methods
- Monitor service status
- Keep credentials secure

Issue 6: Model Performance Issues
Problem: Poor model performance or unexpected results

Symptoms:
- Very low R² scores (< 0.5)
- High prediction errors
- Overfitting (good training, poor test performance)
- Unrealistic predictions

Root Causes:
- Insufficient or poor quality data
- Inappropriate hyperparameters
- Data leakage
- Model complexity mismatch

Diagnostic Steps:
1. Analyze performance metrics:

.. code-block:: python

   print(f"Training MSE: {train_mse:.6f}")
   print(f"Test MSE: {test_mse:.6f}")
   print(f"Training R²: {train_r2:.4f}")
   print(f"Test R²: {test_r2:.4f}")


2. Check for overfitting:

.. code-block:: python

   if train_r2 - test_r2 > 0.2:
   print("Warning: Potential overfitting detected")


3. Analyze prediction errors:

.. code-block:: bash

   residuals = y_test - y_pred
   print(f"Residual statistics:")
   print(f"  Mean: {residuals.mean():.6f}")
   print(f"  Std: {residuals.std():.6f}")
   print(f"  Max: {residuals.max():.6f}")
   print(f"  Min: {residuals.min():.6f}")


Solutions:

Solution 1: Improve Data Quality

.. code-block:: python

   from scipy import stats
   z_scores = np.abs(stats.zscore(data))
   data_clean = data[(z_scores < 3).all(axis=1)]
   if np.array_equal(X_train, X_test):
       print("Warning: Training and test sets are identical!")

Solution 2: Optimize Hyperparameters

.. code-block:: python

   hparam_grid = {
       "dev_tol": [1e-5, 1e-4, 1e-3, 1e-2],
       "feature_tol": [0.1, 0.5, 1.0, 2.0, 5.0],
       "multiple_dev_policy": ["max", "min", "mean", "median"]
   }
   k_folds = 10

Solution 3: Regularize the Model

.. code-block:: python

   def regularized_loss(self, params):
       mse = self._loss(params)
       l2_penalty = 0.01 * np.sum(params[5:]**2)
       return mse + l2_penalty

Solution 4: Use Ensemble Methods

.. code-block:: python

   from sklearn.ensemble import VotingRegressor
   ensemble = VotingRegressor([
       ('dft', dft_model),
       ('linear', LinearRegression()),
       ('ridge', Ridge(alpha=1.0))
   ])
   ensemble.fit(X_train, y_train)

Prevention:
- Always use cross-validation
- Monitor train/test performance gap
- Validate data quality before training
- Use appropriate model complexity

Issue 7: Visualization and Plotting Errors
Problem: Errors when creating plots or visualizations

Symptoms:
- "RuntimeError: Invalid DISPLAY variable"
- "UserWarning: Matplotlib is currently using agg"
- "ValueError: x and y must have same first dimension"
- Blank or empty plots

Root Causes:
- Backend configuration issues
- Data type mismatches
- Missing display (headless environments)
- Memory issues with large datasets

Diagnostic Steps:
1. Check matplotlib backend:

.. code-block:: python

   import matplotlib
   print(f"Backend: {matplotlib.get_backend()}")


2. Test basic plotting:

.. code-block:: python

   import matplotlib.pyplot as plt
   plt.figure()
   plt.plot([1, 2, 3], [1, 4, 2])
   plt.show()


3. Check data compatibility:

.. code-block:: python

   print(f"X shape: {X.shape}")
   print(f"y shape: {y.shape}")
   print(f"X dtype: {X.dtype}")
   print(f"y dtype: {y.dtype}")


Solutions:

Solution 1: Configure Matplotlib Backend

.. code-block:: python

   import matplotlib
   matplotlib.use('Agg')  # headless
   # For Jupyter: %matplotlib inline
   # Interactive: matplotlib.use('TkAgg')

Solution 2: Fix Data Type Issues

.. code-block:: python

   X = X.astype(np.float64)
   y = y.astype(np.float64)
   mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
   X, y = X[mask], y[mask]

Solution 3: Handle Large Datasets

.. code-block:: python

   n_samples = min(1000, len(X))
   indices = np.random.choice(len(X), n_samples, replace=False)
   X_plot, y_plot = X[indices], y[indices]
   plt.scatter(y_plot, y_pred[indices], alpha=0.5, s=1)

Solution 4: Alternative Visualization

.. code-block:: python

   import plotly.express as px
   fig = px.scatter(x=y_test, y=y_pred, title="Model Performance")
   fig.show()
   import seaborn as sns
   sns.scatterplot(x=y_test, y=y_pred)

Prevention:
- Test plotting functionality during setup
- Use appropriate backends for your environment
- Validate data before plotting
- Consider data size for visualization

Issue 8: Cross-Validation Errors
Problem: Errors during k-fold cross-validation

Symptoms:
- "ValueError: n_splits=5 cannot be greater than the number of samples"
- "TypeError: 'NoneType' object is not iterable"
- "MemoryError during cross-validation"

Root Causes:
- Insufficient data for requested folds
- Data splitting issues
- Memory constraints
- Invalid fold configuration

Diagnostic Steps:
1. Check data size:

.. code-block:: python

   print(f"Data size: {len(X)}")
   print(f"Requested folds: {k_folds}")
   if len(X) < k_folds:
   print("Error: Not enough data for requested folds")


2. Test basic splitting:

.. code-block:: python

   from sklearn.model_selection import KFold
   kf = KFold(n_splits=min(k_folds, len(X)//2))
   for train_idx, val_idx in kf.split(X):
   print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")


Solutions:
Solution 1: Adjust Fold Count

.. code-block:: python

   k_folds = min(k_folds, len(X) // 2)
   if k_folds < 2:
       k_folds = 2
   if len(X) < 10:
       from sklearn.model_selection import LeaveOneOut
       cv = LeaveOneOut()
   else:
       from sklearn.model_selection import KFold
       cv = KFold(n_splits=k_folds)

Solution 2: Fix Data Splitting

.. code-block:: python

   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   from sklearn.model_selection import StratifiedKFold
   skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

Solution 3: Reduce Memory Usage
# Use smaller datasets for cross-validation
X_cv = X[:1000]  # Limit to 1000 samples
y_cv = y[:1000]

# Process folds sequentially instead of in parallel
for train_idx, val_idx in kf.split(X_cv):

.. code-block:: bash

   # Process one fold at a time
   pass


Prevention:
- Check data size before cross-validation
- Use appropriate fold counts
- Monitor memory usage
- Test with small datasets first

General Debugging Strategies

1. Enable Verbose Logging
import logging
logging.basicConfig(level=logging.DEBUG)

2. Use Debugging Tools
import pdb
pdb.set_trace()  # Set breakpoint

3. Profile Performance
import cProfile
profiler = cProfile.Profile()
profiler.enable()
# Your code here
profiler.disable()
profiler.print_stats(sort='cumulative')

4. Check System Resources
import psutil
print(f"CPU usage: {psutil.cpu_percent()}%")
print(f"Memory usage: {psutil.virtual_memory().percent}%")
print(f"Disk usage: {psutil.disk_usage('/').percent}%")

5. Validate Inputs
def validate_inputs(X, y):

.. code-block:: bash

   assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
   assert not np.isnan(X).any(), "X contains NaN values"
   assert not np.isnan(y).any(), "y contains NaN values"
   assert not np.isinf(X).any(), "X contains infinite values"
   assert not np.isinf(y).any(), "y contains infinite values"


Getting Help
If problems persist:

1. Check the GitHub issues page
2. Review the API reference for detailed information
3. Consult the usage examples for similar cases
4. Search online for similar error messages
5. Contact the development team with:

.. code-block:: bash

   - Complete error message
   - System information
   - Code snippet that reproduces the issue
   - Expected vs actual behavior


Prevention Strategies
1. Use version control for code and data
2. Test with sample data before full datasets
3. Document your workflow and parameters
4. Keep dependencies updated
5. Use consistent environments
6. Monitor performance metrics
7. Implement proper error handling
8. Regular backups of important work

This troubleshooting guide covers the most common issues encountered when using the DFT Development project. For additional support, please refer to the project repository or contact the development team.
