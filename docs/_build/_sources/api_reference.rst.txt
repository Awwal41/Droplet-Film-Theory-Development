API Reference
=============

Droplet-Film Model Development Project — Technical Documentation.

Overview
--------
This document provides technical documentation for classes, methods, and parameters in the DFT Development project. The API is designed to be both powerful and user-friendly, supporting research and industrial applications.

The project follows object-oriented design with clear separation between physics modeling, data management, and machine learning components.

Core Classes and Modules
-------------------------
The project consists of several key modules:

- **dft_model.py**: Core physics model implementation
- **utils.py**: Data management and utility functions
- Individual Jupyter notebooks for different approaches

DFT Class — Core Physics Model
------------------------------
The DFT class implements the Droplet-Film Model for predicting critical flow rates in gas wells.

Class definition
~~~~~~~~~~~~~~~~

.. code-block:: python

   class DFT:
       """
       Droplet-Film Model for predicting critical flow rates in gas wells.

       This class implements a physics-informed machine learning approach that combines
       fundamental fluid dynamics principles with data-driven optimization to predict
       when gas wells will experience liquid loading.
       """

Constructor
~~~~~~~~~~~

.. code-block:: python

   __init__(self, seed=42, feature_tol=1.0, dev_tol=1e-3, multiple_dev_policy="max")

**Parameters:**

- **seed** (int): Random seed for reproducibility. Default: 42
- **feature_tol** (float): Feature distance threshold for matching. Default: 1.0
- **dev_tol** (float): Deviation tolerance for angle matching. Default: 1e-3
- **multiple_dev_policy** (str): Policy for handling multiple matches. Options: "max", "min", "mean", "median". Default: "max"

**Attributes:** seed, feature_tol, dev_tol, multiple_dev_policy, opt_params (set after fitting), n_train (set after fitting).

Methods — fit
~~~~~~~~~~~~~

.. code-block:: python

   fit(self, X, y)

Train the DFT model on provided data.

**Parameters:** X (np.ndarray) shape (n_samples, 10), y (np.ndarray) shape (n_samples,). **Returns:** self.

**Features (in order):** Dia, Dev(deg), Area (m2), z, GasDens, LiquidDens, g (m/s2), P/T, friction_factor, critical_film_thickness.

**Implementation:** Uses Powell optimization from scipy.optimize; optimizes 5 global parameters (p1–p5) plus alpha per sample; bounds alpha in [0, 1]; max 5000 iterations, 10000 function calls.

Methods — predict
~~~~~~~~~~~~~~~~~

.. code-block:: python

   predict(self, X, dev_train=None, alpha_strategy='enhanced_dev_based')

Make predictions on new data.

**Parameters:** X (np.ndarray), optional dev_train, alpha_strategy (must be 'enhanced_dev_based'). **Returns:** np.ndarray of shape (n_samples,).

**Alpha assignment strategy (by well deviation angle):**

1. Dev < 10°: Regular deviation-based matching

   - Find training samples within dev_tol
   - Apply multiple_dev_policy if multiple matches
   - Use mean training alpha if no matches

2. 10° ≤ Dev < 20°: Minimum alpha strategy

   - Find training samples within dev_tol
   - Use minimum alpha among matches
   - Use mean training alpha if no matches

3. Dev ≥ 20°: Full-feature matching

   - Compute Euclidean distance to all training samples
   - Use closest sample's alpha if distance < feature_tol
   - Use mean training alpha otherwise

Methods — _eq (physics equation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   _eq(self, params, X)

Compute predicted values using the physics equation.

**Physics equation:**

.. math::

   Q_{cr} = p_1 \\sqrt{\\left| \\mathrm{term}_1 \\cdot \\alpha + (1-\\alpha) \\cdot \\mathrm{term}_2 \\right| \\cdot (1/z) \\cdot (P/T) }

Where:

- term1 involves :math:`2 g \\, \\mathrm{Dia}`, :math:`(\\rho_l - \\rho_g)`, :math:`\\cos(\\mathrm{Dev})`, and parameters p4.
- term2 involves :math:`|\\sin(p_5 \\cdot \\mathrm{Dev})|^{p_3}` and :math:`(\\rho_l - \\rho_g)^{p_2} / \\rho_g^2`.

Methods — _loss
~~~~~~~~~~~~~~~

.. code-block:: python

   _loss(self, params)

Compute loss function for optimization. **Returns:** float (MSE).

ChiefBldr Class — Data Management
---------------------------------
The ChiefBldr class handles dataset loading, preprocessing, model training, and evaluation.

Class definition
~~~~~~~~~~~~~~~~

.. code-block:: python

   class ChiefBldr:
       """
       Data management and model training utility class.
       Provides functionality for loading datasets, splitting data,
       training models, and evaluating performance.
       """

Constructor
~~~~~~~~~~~

.. code-block:: python

   __init__(self, path, seed=42, drop_cols=None, includ_cols=None, test_size=0.20, scale=False)

**Parameters:** path (str), seed (int), drop_cols, includ_cols (lists), test_size (float, default 0.20), scale (bool, default False).

**Attributes (set after initialization):** X, y, X_train, X_test, y_train, y_test, feature_names, scaler (if scale=True).

Methods — evolv_model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   evolv_model(self, build_model, hparam_grid, k_folds=5)

Train model with hyperparameter optimization. **Returns:** best trained model. Performs grid search and k-fold cross-validation; stores predictions and metrics (MSE, R²).

QLatticeWrapper Class — Symbolic Regression
-------------------------------------------
Wrapper for Feyn QLattice for automated symbolic regression with a scikit-learn compatible interface.

Constructor
~~~~~~~~~~~

.. code-block:: python

   __init__(self, feature_tags, output_tag="Qcr", seed=42, max_complexity=10, n_epochs=10, criterion="bic")

**Parameters:** feature_tags (List[str]), output_tag, seed, max_complexity, n_epochs, criterion ("bic", "aic", "r2").

**Methods:** fit(X, y), predict(X), express() (returns SymPy expression).

Data Format Requirements
------------------------
Input CSV must contain exactly: Dia, Dev(deg), Area (m2), z, GasDens, LiquidDens, g (m/s2), P/T, friction_factor, critical_film_thickness (types and units as in Installation Guide).

Data Validation, Error Handling, Performance
---------------------------------------------
ChiefBldr validates format, handles missing values and types. The API includes error handling for invalid inputs, missing columns, optimization failures, and (for QLattice) network issues. Memory and training time scale as documented in the guide.

Support and Resources
---------------------
GitHub repository, documentation, community forums, issue tracker, and research papers. For examples, see Usage Examples and the Jupyter notebooks.
