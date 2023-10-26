import numpy as np
import ternary

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
scale = int(1 / step_size)

# Initialize a dictionary to store the entropy values
entropy_dict = {}

# Calculate entropy values
for i in range(scale + 1):
    for j in range(scale - i + 1):
        k = scale - i - j
        p1, p2, p3 = i / scale, j / scale, k / scale
        entropy = calculate_entropy(p1, p2, p3)
        entropy_dict[(i, j, k)] = entropy

# Create a ternary plot
figure, tax = ternary.figure(scale=scale)
tax.heatmap(entropy_dict, style="hexagonal", cmap='viridis')

# Add labels and title
tax.set_title("2D Ternary Plot of Entropy for a Three-Class Problem")
tax.left_axis_label("p1")
tax.right_axis_label("p2")
tax.bottom_axis_label("p3")

# Save the plot as a PNG file
tax.savefig('entropy_3_class_ternary.png')

# Clear the plot
ternary.plt.close()
