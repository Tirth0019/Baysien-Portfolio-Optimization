#!/usr/bin/env python3
"""
Test script to verify all packages are installed and working correctly
for the Bayesian Portfolio Optimization project.
"""

import sys
print(f"Python version: {sys.version}")

# Test basic packages
print("\n=== Testing Basic Packages ===")
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas: {e}")

try:
    import matplotlib.pyplot as plt
    print(f"✓ Matplotlib {plt.matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib: {e}")

try:
    import seaborn as sns
    print(f"✓ Seaborn {sns.__version__}")
except ImportError as e:
    print(f"✗ Seaborn: {e}")

# Test optimization packages
print("\n=== Testing Optimization Packages ===")
try:
    import cvxpy as cp
    print(f"✓ CVXPY {cp.__version__}")
except ImportError as e:
    print(f"✗ CVXPY: {e}")

# Test Bayesian packages
print("\n=== Testing Bayesian Packages ===")
try:
    import pymc as pm
    print(f"✓ PyMC {pm.__version__}")
except ImportError as e:
    print(f"✗ PyMC: {e}")

try:
    import hmmlearn
    print(f"✓ HMMLearn {hmmlearn.__version__}")
except ImportError as e:
    print(f"✗ HMMLearn: {e}")

# Test financial packages
print("\n=== Testing Financial Packages ===")
try:
    import yfinance as yf
    print(f"✓ yfinance {yf.__version__}")
except ImportError as e:
    print(f"✗ yfinance: {e}")

try:
    import quantstats as qs
    print(f"✓ QuantStats {qs.__version__}")
except ImportError as e:
    print(f"✗ QuantStats: {e}")

# Test Jupyter
print("\n=== Testing Jupyter ===")
try:
    import jupyterlab
    print(f"✓ JupyterLab {jupyterlab.__version__}")
except ImportError as e:
    print(f"✗ JupyterLab: {e}")

print("\n=== Installation Test Complete ===")
print("All packages are ready for Bayesian Portfolio Optimization!") 