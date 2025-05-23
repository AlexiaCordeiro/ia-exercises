import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Set up the figure with two subplots side by side
plt.figure(figsize=(18, 6))

# =============================================
# First subplot: 3D Function Visualization
# =============================================
ax1 = plt.subplot(121, projection='3d')

# Generate function data
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = objective_function(np.array([X[i,j], Y[i,j]]))

# Plot the surface
surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('Fitness')
ax1.set_title('Multimodal Function Surface')

# Add colorbar
plt.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

# =============================================
# Second subplot: Convergence Plot (Log Scale)
# =============================================
ax2 = plt.subplot(122)

# Generate some fake convergence data that looks constant
# (In practice, use your actual fitness_history data)
generations = np.arange(MAX_GENERATIONS)
fitness_history = 7.88 - np.exp(-generations/20) + np.random.normal(0, 0.001, MAX_GENERATIONS)

# Plot with logarithmic scale to emphasize small changes
ax2.semilogy(generations, 7.88 - fitness_history, 'r-', linewidth=2)
ax2.set_xlabel('Generation')
ax2.set_ylabel('Distance to Optimum (log scale)')
ax2.set_title('Convergence (Log Scale)')
ax2.grid(True)

# Invert y-axis to show approach to optimum
ax2.invert_yaxis()

# Add horizontal line at theoretical optimum
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.text(MAX_GENERATIONS*0.7, 0.001, 'Theoretical Optimum', 
         verticalalignment='bottom')

plt.tight_layout()
plt.show()
