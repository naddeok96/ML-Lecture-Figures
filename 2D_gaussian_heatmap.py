import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate two skewed Gaussian distributions
x = torch.randn(1000)
y = x + 0.5*torch.randn(1000) + 1  # skewing the y distribution

# 2. Plot the heat map of the distributions
heatmap, xedges, yedges = np.histogram2d(x.numpy(), y.numpy(), bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis')

# 3. Calculate the mean of the distributions
mean_x, mean_y = torch.mean(x).item(), torch.mean(y).item()

# 4. Calculate the radius of the Euclidean ball (circle) around the mean
high_heat_point = (mean_x, mean_y + 1)
radius = 1.5 * torch.dist(torch.tensor([mean_x, mean_y]), torch.tensor(high_heat_point))

# 5. Generate 360 points around the circle and find the highest and lowest heat points
angles = np.linspace(0, 2*np.pi, 360)
heat_values = []

for angle in angles:
    px = mean_x + radius * np.cos(angle)
    py = mean_y + radius * np.sin(angle)
    ix = np.digitize(px, xedges) - 1
    iy = np.digitize(py, yedges) - 1
    heat_values.append(heatmap[ix, iy])

max_heat_idx = np.argmax(heat_values)
min_heat_idx = np.argmin(heat_values)

high_heat_point = (mean_x + radius * np.cos(angles[max_heat_idx]), mean_y + radius * np.sin(angles[max_heat_idx]))
low_heat_point = (mean_x + radius * np.cos(angles[min_heat_idx]), mean_y + radius * np.sin(angles[min_heat_idx]))

# 6. Draw the Euclidean ball (circle) around the mean
circle = plt.Circle((mean_x, mean_y), radius, color='r', fill=False, linestyle='--')
plt.gca().add_patch(circle)

# 7. Add the high heat point (dot) and low heat point (square) to the heat map
plt.scatter(*high_heat_point, c='red', marker='o')
plt.scatter(*low_heat_point, c='blue', marker='s')

# 8. Draw a dotted line from each point to the mean of the distribution
plt.plot([mean_x, high_heat_point[0]], [mean_y, high_heat_point[1]], 'r--')
plt.plot([mean_x, low_heat_point[0]], [mean_y, low_heat_point[1]], 'b--')

# 9. Display the distance value from each point to the mean
dist = torch.dist(torch.tensor([mean_x, mean_y]), torch.tensor(high_heat_point)).item()
plt.text((mean_x + high_heat_point[0]) / 2, (mean_y + high_heat_point[1]) / 2, f'{dist:.2f}', color='white')
plt.text((mean_x + low_heat_point[0]) / 2, (mean_y + low_heat_point[1]) / 2, f'{dist:.2f}', color='white')

# Remove tick marks
plt.xticks([])
plt.yticks([])

# 10. Save the figure
plt.savefig('2D_gaussian_heatmap_w_eucleadian_ball.PNG')