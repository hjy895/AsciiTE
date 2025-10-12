#!/usr/bin/env python3
"""
Test script to check if the environment is working
"""
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"PyTorch not available: {e}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"Transformers not available: {e}")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"Pandas not available: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy not available: {e}")

try:
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"Scikit-learn not available: {e}")

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"Matplotlib not available: {e}")

print("Environment test completed!")







