Installation Guide
==================

Droplet-Film Model Development Project
Complete Setup Instructions

System Requirements
Before installing the DFT Development project, ensure your system meets the following requirements:

Minimum System Requirements
- Python 3.7 or higher (Python 3.9+ recommended)
- 8GB RAM (16GB recommended for large datasets)
- 2GB free disk space
- Internet connection for package installation
- Operating System: Windows 10/11, macOS 10.14+, or Ubuntu 18.04+

Recommended System Configuration
- Python 3.9 or 3.10
- 16GB RAM or more
- 5GB free disk space
- SSD storage for better performance
- Multi-core processor for faster training

Python Environment Setup
The project requires a properly configured Python environment. We strongly recommend using a virtual environment to avoid conflicts with other projects.

Step 1: Install Python
Download and install Python from the official website (python.org):
- Choose Python 3.9 or 3.10 for best compatibility
- Ensure "Add Python to PATH" is checked during installation
- Verify installation by opening a terminal and running: python --version

Step 2: Create Virtual Environment
Create a dedicated virtual environment for the project:

.. code-block:: bash

   # Windows
   python -m venv dft_env
   dft_env\\Scripts\\activate
   # macOS/Linux
   python3 -m venv dft_env
   source dft_env/bin/activate

Step 3: Upgrade pip

.. code-block:: bash

   python -m pip install --upgrade pip

Package Installation Methods
The project can be installed using several methods. Choose the one that best fits your needs:

Method 1: Complete Installation (Recommended)

.. code-block:: bash

   git clone https://github.com/your-username/Droplet-Film-Theory-Development.git
   cd Droplet-Film-Theory-Development
   pip install -e .

Method 2: Manual Installation

.. code-block:: bash

   pip install numpy>=1.21.0 pandas>=1.3.0 scipy>=1.7.0 scikit-learn>=1.0.0
   pip install matplotlib>=3.4.0 seaborn>=0.11.0 plotly>=5.0.0
   pip install jupyter notebook ipywidgets
   pip install feyn pysindy xgboost

Method 3: Using Requirements File

.. code-block:: bash

   pip install -r requirements.txt
   # Or create one: pip freeze > requirements.txt

Verification and Testing

Step 1: Test Basic Imports

.. code-block:: bash

   python -c "import numpy; print('NumPy version:', numpy.__version__)"
   python -c "import pandas; print('Pandas version:', pandas.__version__)"
   python -c "import scipy; print('SciPy version:', scipy.__version__)"
   python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"

Step 2: Test Project-Specific Imports

.. code-block:: bash

   python -c "from scripts_2.0.dft_model import DFT; print('DFT import successful')"
   python -c "from scripts_2.0.utils import ChiefBldr; print('ChiefBldr import successful')"
   python -c "from scripts_2.0.utils import QLatticeWrapper; print('QLatticeWrapper import successful')"

Step 3: Run Basic Functionality Test

.. code-block:: python

   import numpy as np
   from scripts_2.0.dft_model import DFT
   X = np.random.rand(10, 10)
   y = np.random.rand(10)
   model = DFT(seed=42)
   model.fit(X, y)
   model.predict(X)
   print('DFT model test successful')

Jupyter Notebook Setup
For interactive development and exploration:

Step 1: Install Jupyter
pip install jupyter notebook ipywidgets

Step 2: Launch Jupyter
jupyter notebook

Step 3: Navigate to Project
Open the scripts_2.0 directory and explore the available notebooks:
- DFT-PISR.ipynb: Main physics-informed implementation
- xgboost.ipynb: Gradient boosting approach
- sindy.ipynb: Symbolic regression analysis
- QLattice.ipynb: Automated model discovery

Data Preparation
Before using the project, prepare your dataset:

Required Data Format
Your data must be in CSV format with the following columns:
- Dia: Well diameter (meters)
- Dev(deg): Well deviation angle (degrees)
- Area (m2): Cross-sectional area (square meters)
- z: Elevation change (meters)
- GasDens: Gas density (kg/m³)
- LiquidDens: Liquid density (kg/m³)
- g (m/s2): Gravitational acceleration (m/s²)
- P/T: Pressure/Temperature ratio (Pa/K)
- friction_factor: Friction factor (dimensionless)
- critical_film_thickness: Critical film thickness (meters)

Data Validation
Use the following code to validate your data:

import pandas as pd
import numpy as np

def validate_data(file_path):

.. code-block:: python

   data = pd.read_csv(file_path)
   required_cols = ['Dia', 'Dev(deg)', 'Area (m2)', 'z',
   'GasDens', 'LiquidDens', 'g (m/s2)',
   'P/T', 'friction_factor', 'critical_film_thickness']
   
   missing_cols = [col for col in required_cols if col not in data.columns]
   if missing_cols:
       print(f"Missing columns: {missing_cols}")
       return False
   
   print("All required columns present")
   print(f"Data shape: {data.shape}")
   print(f"Data types:\n{data.dtypes}")
   
   return True


# Validate your data
validate_data("your_data.csv")

Common Installation Issues
Here are solutions to frequently encountered problems:

Issue 1: Permission Errors — use ``pip install --user package_name`` or a virtual environment.

Issue 2: Version Conflicts — use a fresh venv:

.. code-block:: bash

   python -m venv fresh_env
   fresh_env\\Scripts\\activate
   pip install package_name

Issue 3: Memory Issues
Problem: Out of memory during installation
Solution: Close other applications and try again
- Close unnecessary programs
- Restart your computer
- Use pip install --no-cache-dir package_name

Issue 4: Network Issues

.. code-block:: bash

   pip install -i https://pypi.org/simple/ package_name
   pip install --trusted-host pypi.org --trusted-host pypi.python.org package_name

Issue 5: Compilation Errors
Problem: C/C++ compilation errors on Windows
Solution: Install Microsoft Visual C++ Build Tools
- Download from Microsoft website
- Install "C++ build tools" workload
- Restart terminal and try again

Issue 6: QLattice Connection Issues
Problem: Cannot connect to Feyn QLattice
Solution: Check network and credentials
- Ensure internet connection
- Verify Feyn account setup
- Check firewall settings
- Try alternative symbolic regression (PySINDy)

Development Environment Setup

Code quality and testing:

.. code-block:: bash

   pip install black flake8 pytest mypy
   black scripts_2.0/*.py
   flake8 scripts_2.0/*.py
   mypy scripts_2.0/*.py
   pytest tests/

IDE Configuration (Visual Studio Code): add to ``settings.json``:

.. code-block:: json

   {
       "python.defaultInterpreterPath": "./dft_env/Scripts/python.exe",
       "python.linting.enabled": true,
       "python.linting.flake8Enabled": true,
       "python.formatting.provider": "black"
   }

PyCharm: set project interpreter to the virtual environment, enable code inspection, and use PEP 8 code style.

Performance Optimization

.. code-block:: bash

   pip install numpy --upgrade pandas --upgrade scipy --upgrade
   # Or with Conda: conda install numpy pandas scipy scikit-learn matplotlib

Install Intel MKL Libraries
pip install mkl

Security Considerations
- Always use virtual environments
- Keep packages updated regularly
- Use trusted package sources
- Verify package signatures when possible
- Never install packages as root/admin unless necessary

Maintenance and Updates

.. code-block:: bash

   pip install --upgrade package_name
   pip list --outdated
   pip audit
   pip freeze > requirements_backup.txt
   pip cache purge

Uninstallation

.. code-block:: bash

   # Windows: rmdir /s dft_env
   # macOS/Linux: rm -rf dft_env
   pip uninstall package_name
   pip cache purge

Installation Checklist
Use this checklist to verify your installation:

□ Python 3.7+ installed and accessible
□ Virtual environment created and activated
□ Core packages installed (numpy, pandas, scipy, sklearn)
□ Visualization packages installed (matplotlib, seaborn)
□ Jupyter notebook working
□ Project modules importing successfully
□ Basic functionality test passing
□ Sample data prepared and validated
□ Jupyter notebooks opening correctly

Next Steps
After successful installation:

1. Read the main documentation for project overview
2. Explore the usage examples for practical implementation
3. Review the API reference for technical details
4. Check the troubleshooting guide for common issues
5. Start with the DFT-PISR.ipynb notebook
6. Prepare your own dataset for analysis

Support and Resources
If you encounter issues:

1. Check this installation guide first
2. Review the troubleshooting section
3. Search the GitHub issues
4. Contact the development team
5. Join the community discussions

The project is actively maintained and supported by the development team and community contributors.
