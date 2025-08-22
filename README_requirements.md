# Python Requirements for DFT Project

This project uses a single, comprehensive requirements file that covers all use cases.

## 📁 **Requirements File**

### `requirements.txt` - Complete Requirements
- **Use case**: All project functionality including enhanced plotting
- **Contains**: Core libraries + optional advanced features + development tools
- **When to use**: Always - this is the only requirements file you need

## 🚀 **Installation Instructions**

### Quick Start (Recommended)
```bash
pip install -r requirements.txt
```

### Alternative: Install with Conda
```bash
conda install --file requirements.txt
```

### For Virtual Environment (Best Practice)
```bash
# Create virtual environment
python -m venv dft_env

# Activate (Windows)
dft_env\Scripts\activate

# Activate (macOS/Linux)
source dft_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

## 🔧 **Package Categories**

### Core Required Packages (Always Installed)
- **numpy**: Numerical computing and array operations
- **pandas**: Data manipulation and analysis
- **scipy**: Scientific computing and optimization
- **scikit-learn**: Machine learning algorithms and utilities
- **matplotlib**: Basic plotting library
- **seaborn**: Statistical data visualization
- **jupyter**: Jupyter notebook environment
- **ipykernel**: Python kernel for Jupyter

### Optional Packages (Commented Out)
To use these, uncomment the lines in `requirements.txt`:

#### Advanced Machine Learning
```bash
# Uncomment in requirements.txt:
# xgboost>=1.5.0
# lightgbm>=3.3.0
# catboost>=1.0.0
```

#### Symbolic Regression
```bash
# Uncomment in requirements.txt:
# pysindy>=0.4.0
# feyn>=3.0.0
```

#### Enhanced Plotting
```bash
# Uncomment in requirements.txt:
# plotly>=5.0.0
# bokeh>=2.4.0
# colorcet>=3.0.0
# palettable>=3.3.0
```

#### Development Tools
```bash
# Uncomment in requirements.txt:
# pytest>=6.0.0
# black>=22.0.0
# flake8>=4.0.0
```

## 🐍 **Python Version Compatibility**

- **Python 3.8+**: All packages work natively
- **Python 3.7**: Most packages work, some may need older versions
- **Python 3.6**: Limited compatibility, not recommended
- **Python 3.5 and below**: Not supported

## 🔍 **Troubleshooting Common Issues**

### Issue 1: Package Version Conflicts
```bash
# Create a fresh virtual environment
python -m venv dft_env
source dft_env/bin/activate  # On Windows: dft_env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue 2: Missing Dependencies
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv

# Install system dependencies (macOS)
brew install python3
```

### Issue 3: Permission Errors
```bash
# Use user installation
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv dft_env
source dft_env/bin/activate
pip install -r requirements.txt
```

## 📊 **Package Version Matrix**

| Package | Minimum Version | Recommended Version | Notes |
|---------|----------------|-------------------|-------|
| numpy | 1.21.0 | 1.24.0+ | Core dependency |
| pandas | 1.3.0 | 2.0.0+ | Data handling |
| scikit-learn | 1.0.0 | 1.3.0+ | ML algorithms |
| matplotlib | 3.5.0 | 3.7.0+ | Plotting |
| seaborn | 0.11.0 | 0.12.0+ | Statistical plots |

## 🎯 **Quick Start for Different Use Cases**

### For Data Scientists
```bash
pip install -r requirements.txt
jupyter notebook
```

### For Developers
```bash
# Uncomment development tools in requirements.txt first
pip install -r requirements.txt
```

### For Plotting Only
```bash
# Uncomment enhanced plotting packages in requirements.txt first
pip install -r requirements.txt
```

### For Production Deployment
```bash
# Keep only core packages, comment out optional ones
pip install -r requirements.txt
```

## 🔄 **Updating Requirements**

### Check for Updates
```bash
pip list --outdated
```

### Update Specific Package
```bash
pip install --upgrade package_name
```

### Update All Packages
```bash
pip install --upgrade -r requirements.txt
```

### Generate Current Requirements
```bash
pip freeze > requirements_current.txt
```

## 📝 **Notes**

1. **Single File**: All requirements are now in one file for simplicity
2. **Virtual Environments**: Always use virtual environments for project isolation
3. **Optional Packages**: Uncomment only what you need
4. **Version Pinning**: Consider pinning exact versions for production
5. **Security**: Regularly update packages for security patches

## 🆘 **Getting Help**

If you encounter issues:
1. Check Python version compatibility
2. Verify virtual environment setup
3. Check system dependencies
4. Review package version conflicts
5. Consult package documentation
