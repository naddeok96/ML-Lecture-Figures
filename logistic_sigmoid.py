import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LogisticSigmoid(nn.Module):
    """
    A simple logistic sigmoid model for binary classification.
    """
    def __init__(self, lr=0.1, w0=-0.02, w1=0.06, w2=-0.05):
        """
        Initialize the model with hyperparameters and weights.
        """
        super(LogisticSigmoid, self).__init__()

        # Define hyperparameters
        self.lr = lr

        # Define the learnable parameters (weights)
        self.weights = torch.tensor([w0, w1, w2])

    def forward(self, x1, x2):
        """
        Forward pass to compute the sigmoid output.
        """
        # Define the operations in the forward pass with labels
        inputs = torch.tensor([1.0, x1, x2])
        d = torch.sum(self.weights * inputs)
        e = -d
        f = torch.exp(e)
        g = f + 1
        h = 1 / g
        
        # Store variables for later use in backward pass
        self.e = e
        self.g = g
        self.inputs = inputs

        return h

    def calculate_loss(self, pred, label):
        """
        Calculate the squared error loss.
        """
        # Calculate squared error
        i = pred - label
        j = i ** 2

        # Store variables for later use in backward pass
        self.i = i

        return j

    def backward(self):
        """
        Backward pass to compute gradients.
        """
        # Compute the gradient using matrix multiplication
        delta = -2 * self.i * (torch.exp(self.e) / (self.g ** 2))
        grad_wrt_weights = delta * self.inputs

        # Store the gradient for later use in optimization step
        self.grad_wrt_weights = grad_wrt_weights 

        return grad_wrt_weights

    def step(self):
        """
        Update the weights using the computed gradients.
        """
        self.weights = self.weights + self.lr * self.grad_wrt_weights

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 1000
    start_lr = 0.1
    end_lr = 0.001
    num_starts = 50

    # Given data points
    x1 = torch.tensor(0.42)
    x2 = torch.tensor(0.65)
    label = torch.tensor(0.05)

    # Lists to store loss values for each start
    all_loss_values = []
    lr_rates = []

    for start in range(num_starts):
        # Randomly initialize weights
        w0 = torch.rand(1).item()
        w1 = torch.rand(1).item()
        w2 = torch.rand(1).item()

        model = LogisticSigmoid(lr=start_lr, w0=w0, w1=w1, w2=w2)

        # List to store loss values for this start
        loss_values = []

        for i in range(num_epochs):
            result = model(x1, x2)
            loss = model.calculate_loss(result, label)

            model.backward()
            model.step()

            # Append loss values to the list
            loss_values.append(loss.item())

            print(f"Start {start + 1}, Epoch: {i+1}, Loss: {loss.item()}, Pred: {result}, Weights: {model.weights}")

            # Anneal the learning rate
            model.lr = 10 ** (np.log10(start_lr) + i * (np.log10(end_lr) - np.log10(start_lr)) / (num_epochs - 1))
            if start == 0:
                lr_rates.append(model.lr)

        # Append loss values for this start to the list of all loss values
        all_loss_values.append(loss_values)

    # Plotting
    plt.figure(figsize=(16, 6))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    for start in range(num_starts):
        plt.plot(np.arange(num_epochs), all_loss_values[start], label=f'Loss (Start {start + 1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch (All Starts)')
    plt.legend()

    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(num_epochs), lr_rates, label='Learning Rate', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs. Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('combined_loss_lr_plot.png')
