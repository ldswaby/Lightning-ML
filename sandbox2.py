import numpy as np

from src.lightning_ml.datasets import NumpyLabelledDataset

# Set random seed for reproducibility
np.random.seed(42)

# Create 100 samples with 2 features each
X = np.random.randn(100, 2)

# Create labels based on a linear decision boundary (just for demo)
# Label = 1 if x1 + x2 > 0, else 0
y = (X[:, 0] + X[:, 1] > 0).astype(int)


dataset = NumpyLabelledDataset(X, y)
