import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Function to calculate entropy for a three-class problem
def calculate_entropy(p1, p2, p3):
    entropy = 0
    if p1 > 0:
        entropy += -p1 * np.log2(p1)
    if p2 > 0:
        entropy += -p2 * np.log2(p2)
    if p3 > 0:
        entropy += -p3 * np.log2(p3)
    return entropy

# Generate probabilities for p1, p2, and p3 such that p1 + p2 + p3 = 1
step_size = 0.01
p1_values = np.arange(0, 1 + step_size, step_size)
p2_values = np.arange(0, 1 + step_size, step_size)

# Initialize lists to store p1, p2, p3, and entropy values for plotting
data_list = []

# Calculate entropy values
for p1 in p1_values:
    for p2 in p2_values:
        p3 = 1 - p1 - p2
        if p3 >= 0:
            entropy = calculate_entropy(p1, p2, p3)
            data_list.append([p1, p2, p3, entropy])

# Convert list to numpy array
data_array = np.array(data_list)

# Perform PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_array[:, :3])

# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'})

# Plot original data
sc = axes[0, 0].scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2], c=data_array[:, 3], cmap='viridis')
axes[0, 0].set_title('Original Data')
axes[0, 0].set_xlabel('p1')
axes[0, 0].set_ylabel('p2')
axes[0, 0].set_zlabel('p3')

# Plot PCA angles
for i in range(3):
    ax = axes.flatten()[i + 1]
    sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=data_array[:, 3], cmap='viridis')
    ax.view_init(elev=30, azim=45 + i * 45)
    ax.set_title(f'PCA Angle {i + 1}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

# Add color bar
plt.colorbar(sc, ax=axes.ravel().tolist())

# Save the plot as a PNG file
fig.savefig('entropy_3_class_pca.png')
