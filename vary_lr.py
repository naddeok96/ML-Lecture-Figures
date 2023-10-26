import numpy as np
import matplotlib.pyplot as plt

def loss(x):
    """Quadratic loss function."""
    return x**2

def gradient(x):
    """Gradient of the function f(x) = x^2."""
    return 2*x

def sgd_update(initial_x, learning_rate, num_steps):
    """Performs SGD updates and returns both x values and corresponding loss values."""
    x = initial_x
    x_history = [x]
    loss_history = [loss(x)]
    for _ in range(num_steps):
        x = x - learning_rate * gradient(x)
        x_history.append(x)
        loss_history.append(loss(x))
    return x_history, loss_history

def plot_sgd_steps_on_loss_curve():
    initial_x = 10.0
    num_steps = 10
    
    lr_values = [0.01, 1.5] # Small and large learning rates
    colors = ['r', 'b']
    labels = ['lr = 0.01', 'lr = 1.5']

    # Generate continuous data for the loss curve
    x_values = np.linspace(-initial_x-5, initial_x+5, 400)
    y_values = loss(x_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, 'k-', label='f(x) = x^2', linewidth=1.5)

    # Plot SGD steps on the loss curve
    for lr, color, label in zip(lr_values, colors, labels):
        x_history, loss_history = sgd_update(initial_x, lr, num_steps)
        plt.scatter(x_history, loss_history, color=color, marker='o')
        plt.plot(x_history, loss_history, color=color, linestyle='--', linewidth=1.2, label=label)

    plt.title("SGD Steps on Convex Loss Curve for Different Learning Rates")
    plt.xlabel("Value of x")
    plt.ylabel("Loss f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lr_variation.png')
    # plt.show()

# Execute the plotting function
plot_sgd_steps_on_loss_curve()
