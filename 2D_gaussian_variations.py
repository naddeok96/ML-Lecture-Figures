import numpy as np
import matplotlib.pyplot as plt

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

# Create grid and multivariate normal
x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
x, y = np.meshgrid(x, y)
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y

# Gaussian 1
mu1 = np.array([-2, -2])
Sigma1 = np.array([[1, 0.5], [0.5, 1]])
g1 = multivariate_gaussian(pos, mu1, Sigma1)

# Gaussian 2
mu2 = np.array([2, 2])
Sigma2 = np.array([[1.5, -0.5], [-0.5, 0.5]])
g2 = multivariate_gaussian(pos, mu2, Sigma2)

# Make the plot
plt.figure(figsize=(8, 6))
plt.imshow(g1 + g2, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis')
plt.colorbar()
plt.title("Two 2D Gaussian Variations with Covariance")
plt.savefig("2d_gaussian_variations.PNG")
