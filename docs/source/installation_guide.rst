Installation Guide
==================

Droplet-Film Model Development Project — Complete Setup Instructions.

System Requirements
-------------------

Before installing, ensure your system meets the following requirements.

Minimum
~~~~~~~

- Python 3.7 or higher (3.9+ recommended)
- 8 GB RAM (16 GB recommended for large datasets)
- 2 GB free disk space
- Internet connection for package installation
- OS: Windows 10/11, macOS 10.14+, or Ubuntu 18.04+

Recommended
~~~~~~~~~~~

- Python 3.9 or 3.10
- 16 GB RAM or more
- 5 GB free disk space
- SSD storage
- Multi-core processor


Python Environment Setup
------------------------

Use a virtual environment to avoid conflicts with other projects.

Step 1 — Install Python
~~~~~~~~~~~~~~~~~~~~~~~~

- Download from https://www.python.org/
- Choose Python 3.9 or 3.10 for best compatibility
- During installation, check **Add Python to PATH**
- Verify: ``python --version`` (or ``python3 --version`` on macOS/Linux)

Step 2 — Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Windows
   python -m venv dft_env
   dft_env\Scripts\activate

   # macOS/Linux
   python3 -m venv dft_env
   source dft_env/bin/activate

Step 3 — Upgrade pip
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install --upgrade pip


Package Installation
--------------------

Choose one of the following methods.

Method 1: From repository (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/your-username/Droplet-Film-Model-Development.git
   cd Droplet-Film-Model-Development
   pip install -e .

Method 2: Manual install
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install numpy pandas scipy scikit-learn
   pip install matplotlib seaborn plotly
   pip install jupyter notebook ipywidgets
   pip install feyn pysindy xgboost

Method 3: Requirements file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt


Verification
------------

Step 1 — Test basic imports
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -c "import numpy; print('NumPy:', numpy.__version__)"
   python -c "import pandas; print('Pandas:', pandas.__version__)"
   python -c "import scipy; print('SciPy:', scipy.__version__)"
   python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"

Step 2 — Test project modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run from the **project root** so ``src`` and ``models`` are on the path:

.. code-block:: bash

   python -c "from src.dfm_src import DFT; print('DFT OK')"
   python -c "from models.utils import Helm; print('Helm OK')"
   python -c "from models.utils import QLatticeWrapper; print('QLatticeWrapper OK')"

Step 3 — Quick functionality test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from src.dfm_src import DFT

   X = np.random.rand(10, 10)
   y = np.random.rand(10)
   model = DFT(seed=42)
   model.fit(X, y)
   model.predict(X)
   print("DFT model test successful")


Jupyter Notebooks
-----------------

Step 1 — Install Jupyter
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install jupyter notebook ipywidgets

Step 2 — Launch and open notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   jupyter notebook

Then open the project root or ``models/`` and run:

- **DFT-PISR.ipynb** — physics-informed DFM
- **xgboost.ipynb** — gradient boosting
- **sindy.ipynb** — symbolic regression (PySINDy)
- **QLattice.ipynb** — QLattice symbolic regression


Data Preparation
----------------

Required CSV columns
~~~~~~~~~~~~~~~~~~~~

Your CSV must include these 10 feature columns (and for **Helm**, also **Qcr**, **Gasflowrate**, **Test status**):

- **Dia** — well diameter (m)
- **Dev(deg)** — well deviation angle (degrees)
- **Area (m2)** — cross-sectional area (m²)
- **z** — elevation change (m)
- **GasDens** — gas density (kg/m³)
- **LiquidDens** — liquid density (kg/m³)
- **g (m/s2)** — gravitational acceleration (m/s²)
- **P/T** — pressure/temperature ratio (Pa/K)
- **friction_factor** — dimensionless
- **critical_film_thickness** — critical film thickness (m)

Validate your data
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd

   def validate_data(file_path):
       data = pd.read_csv(file_path)
       required = [
           'Dia', 'Dev(deg)', 'Area (m2)', 'z', 'GasDens', 'LiquidDens',
           'g (m/s2)', 'P/T', 'friction_factor', 'critical_film_thickness'
       ]
       missing = [c for c in required if c not in data.columns]
       if missing:
           print("Missing columns:", missing)
           return False
       print("All required columns present. Shape:", data.shape)
       return True

   validate_data("your_data.csv")


Common Installation Issues
--------------------------

Permission errors
~~~~~~~~~~~~~~~~~

Use a virtual environment, or: ``pip install --user package_name``

Version conflicts
~~~~~~~~~~~~~~~~~

Use a fresh virtual environment and reinstall:

.. code-block:: bash

   python -m venv fresh_env
   # Windows: fresh_env\Scripts\activate
   # macOS/Linux: source fresh_env/bin/activate
   pip install package_name

Memory errors
~~~~~~~~~~~~~

- Close other applications
- Use: ``pip install --no-cache-dir package_name``
- Restart and try again

Network / download failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -i https://pypi.org/simple/ package_name
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package_name

Compilation errors (Windows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install **Microsoft Visual C++ Build Tools** (include “Desktop development with C++” or “C++ build tools”), then restart the terminal.

QLattice connection
~~~~~~~~~~~~~~~~~~~

- Check internet and firewall
- Verify Feyn account and API setup
- Use PySINDy or other methods if QLattice is unavailable


Development Setup
-----------------

Code quality and tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install black flake8 pytest mypy
   black src/*.py models/*.py
   flake8 src/*.py models/*.py
   mypy src/*.py models/*.py
   pytest tests/

IDE (VS Code)
~~~~~~~~~~~~~

Add to ``.vscode/settings.json`` or user settings:

.. code-block:: json

   {
       "python.defaultInterpreterPath": "${workspaceFolder}/dft_env/Scripts/python.exe",
       "python.linting.enabled": true,
       "python.linting.flake8Enabled": true,
       "python.formatting.provider": "black"
   }

On macOS/Linux, use ``dft_env/bin/python`` instead of ``Scripts/python.exe``.

PyCharm: set the project interpreter to the virtual environment and enable PEP 8 inspection.


Performance and Maintenance
---------------------------

Upgrade packages
~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install --upgrade package_name
   pip list --outdated
   pip audit

Optional: Intel MKL
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install mkl

Backup and cleanup
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip freeze > requirements_backup.txt
   pip cache purge


Uninstall
---------

.. code-block:: bash

   # Remove virtual environment
   # Windows: rmdir /s dft_env
   # macOS/Linux: rm -rf dft_env

   pip uninstall package_name
   pip cache purge


Installation Checklist
----------------------

- [ ] Python 3.7+ installed and on PATH
- [ ] Virtual environment created and activated
- [ ] Core packages installed (numpy, pandas, scipy, scikit-learn)
- [ ] Optional: matplotlib, seaborn, jupyter
- [ ] Project imports work (``from src.dfm_src import DFT``, ``from models.utils import Helm``)
- [ ] Quick DFT test runs successfully
- [ ] CSV data has required columns (and Qcr, Gasflowrate, Test status for Helm)


Next Steps
----------

1. See **Usage examples** for the DFT and other model tutorials.
2. See **API reference** for classes and parameters.
3. See **Troubleshooting** if you run into errors.
4. Open **DFT-PISR.ipynb** and run the workflow end-to-end.

For more help: check the troubleshooting guide, search GitHub issues, or contact the development team.
