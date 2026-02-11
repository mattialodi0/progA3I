import torch
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib
from utils import get_gaussian_lattice

# --- Usage Example ---
# 1. Get the lattice
z_fixed, point_colors = get_gaussian_lattice()

# 3. Visualize
plt.scatter(z_fixed[:, 0], z_fixed[:, 1], c=point_colors, s=50, edgecolors='k', linewidth=0.5)
plt.show()