import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Function to calculate the Maximum Likelihood Estimation (MLE) for a 2D Gaussian
def calculate_mle(data, mean, cov):
    rv = multivariate_normal(mean, cov)
    likelihoods = rv.pdf(data)
    return np.sum(np.log(likelihoods))

# Generate synthetic data for the "True" Gaussian distribution
mean_true = [0, 0]
cov_true = [[1, 0.5], [0.5, 1]]
data = np.random.multivariate_normal(mean_true, cov_true, 1000)

# Parameters for the "Bad Guess" Gaussian distribution
mean_bad_guess = [5, 5]
cov_bad_guess = [[3, 5], [1, 5]]

# Calculate MLE for each Gaussian
mle_true = calculate_mle(data, mean_true, cov_true)
mle_bad_guess = calculate_mle(data, mean_bad_guess, cov_bad_guess)

# Create a grid for plotting
x, y = np.mgrid[-8:8:.01, -8:8:.01]
pos = np.dstack((x, y))

# Create the Gaussian objects
rv_true = multivariate_normal(mean_true, cov_true)
rv_bad_guess = multivariate_normal(mean_bad_guess, cov_bad_guess)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the "True" Gaussian
axes[0].contourf(x, y, rv_true.pdf(pos))
axes[0].scatter(data[:, 0], data[:, 1], c='red', s=10)
axes[0].set_title(f"True Gaussian\nMLE = {mle_true}")

# Plot the "Bad Guess" Gaussian
axes[1].contourf(x, y, rv_bad_guess.pdf(pos))
axes[1].scatter(data[:, 0], data[:, 1], c='red', s=10)
axes[1].set_title(f"Bad Guess\nMLE = {mle_bad_guess}")

# Save the plot
plt.savefig("mle_2d_gaussian.PNG")
