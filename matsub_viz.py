import matplotlib.pyplot as plt
import numpy as np

def draw_vector(ax, dim, x_offset=0, y_offset=0, label=None, color='blue'):
    m, n = dim
    for i in range(m):
        for j in range(n):
            ax.add_patch(plt.Rectangle((j + x_offset, m - i - 1 + y_offset), 1, 1, edgecolor='black', facecolor=color))
    if label:
        ax.text(x_offset + n/2, m + y_offset + 0.5, label, ha='center', fontsize=16)

def elementwise_subtraction(dim1, dim2):
    m1, n1 = dim1
    m2, n2 = dim2
    if m1 != m2 or n1 != n2:
        raise ValueError("Matrix dimensions do not match for element-wise subtraction")
    return dim1  # Resulting matrix has the same dimensions

def illustrate_elementwise_subtraction(dim1, dim2):
    result_dim = elementwise_subtraction(dim1, dim2)
    
    total_width = dim1[1] + dim2[1] + result_dim[1] + 3
    total_height = max(dim1[0], dim2[0], result_dim[0]) + 3
    
    fig, ax = plt.subplots(figsize=(total_width, total_height))
    ax.axis('off')
    
    center_offset = (total_height - max(dim1[0], dim2[0], result_dim[0])) / 2
    
    # Draw the first vector/matrix
    draw_vector(ax, dim1, y_offset=center_offset, label=f"{dim1[0]}x{dim1[1]}", color='blue')
    
    # Draw the minus sign
    ax.text(dim1[1] + 0.5, total_height/2, '-', ha='center', va='center', fontsize=20)
    
    # Draw the second vector/matrix
    draw_vector(ax, dim2, x_offset=dim1[1] + 1, y_offset=center_offset, label=f"{dim2[0]}x{dim2[1]}", color='green')
    
    # Draw the equal sign
    ax.text(dim1[1] + dim2[1] + 1.5, total_height/2, '=', ha='center', va='center', fontsize=20)
    
    # Draw the resulting vector/matrix
    draw_vector(ax, result_dim, x_offset=dim1[1] + dim2[1] + 2, y_offset=center_offset, label=f"{result_dim[0]}x{result_dim[1]}", color='red')
    
    # Adjust the axis limits to center the drawing
    ax.set_xlim(-1, total_width + 1)
    ax.set_ylim(-1, total_height + 1)
    
    # Save the figure
    filename = f"sub_{dim1[0]}{dim1[1]}_{dim2[0]}{dim2[1]}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300, pad_inches=0.1)

dim1 = (3, 1)
dim2 = (3, 1)
illustrate_elementwise_subtraction(dim1, dim2)
