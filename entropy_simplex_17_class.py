import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Function to calculate entropy for a 5-class problem
def calculate_entropy(probabilities):
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

# Initialize probabilities and calculate entropy for each vertex
num_classes = 5
angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
x = np.cos(angles)
y = np.sin(angles)

# Create a grid within the pentagon
corners = np.array([[x[i], y[i]] for i in range(num_classes)])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
refiner = tri.UniformTriRefiner(triangle)
triang, z = refiner.refine_triangulation(subdiv=4, z=None)

# Calculate entropy for each point in the grid
entropies = []
for coords in triang.x, triang.y:
    barycentric_coords = np.linalg.solve(
        np.array([[x[0], x[1], x[2]], [y[0], y[1], y[2]], [1, 1, 1]]),
        np.array([coords[0], coords[1], 1])
    )
    probabilities = np.zeros(num_classes)
    probabilities[:3] = barycentric_coords
    probabilities[3] = 1 - np.sum(probabilities[:3])
    entropies.append(calculate_entropy(probabilities))

# Plot the pentagon
plt.figure(figsize=(10, 10))
plt.axis("equal")
plt.tricontourf(triang, entropies, levels=50, cmap="viridis")
plt.colorbar(label="Entropy")
plt.title("5-Class Problem Entropy Pentagon")
plt.xlabel("x")
plt.ylabel("y")

# Save the plot as a PNG file
plt.savefig('entropy_5_class_pentagon.png')

# Show the plot (optional)
# plt.show()
