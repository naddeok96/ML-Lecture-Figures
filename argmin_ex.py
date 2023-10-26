import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x)
def f(x):
    return x**2 + 3

# Generate x values
x = np.linspace(-10, 10, 400)

# Generate y values
y = f(x)

# Plot the function
plt.plot(x, y, label="f(x) = x^2 + 3")

# Plot the point for f(0)
plt.scatter([0], [f(0)], color='red', zorder=5)  # zorder ensures the point is drawn on top
plt.text(0, f(0) + 0.5, 'min', horizontalalignment='center', color='red')
plt.text(0, f(0) - 1, 'x=0 is the arg min', horizontalalignment='center', color='red')

# Setting title and labels
plt.title("Plot of f(x) = x^2 + 3")
plt.xlabel("x")
plt.ylabel("f(x)")

# Adjust y-axis
plt.ylim(-1, 15)

# Display the legend
plt.legend()

# Save the plot
plt.savefig("argmin_ex.png")