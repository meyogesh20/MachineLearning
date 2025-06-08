import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_svm_boundary_3d(model, A, b):
    # Ensure inputs are numpy arrays
    X = np.asarray(A)
    y = np.asarray(b)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], 0, c=y, cmap='seismic', s=50, alpha=1)

    # Create grid to evaluate model
    xlim = [X[:, 0].min() - 1, X[:, 0].max() + 1]
    ylim = [X[:, 1].min() - 1, X[:, 1].max() + 1]
    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    # To suppress warning about Feature names while plotting the graph
    if hasattr(A, "columns"):
        xy = pd.DataFrame(xy, columns=A.columns)
    
    
    Z = model.decision_function(xy).reshape(XX.shape)

    # Plot the decision surface
    ax.plot_surface(XX, YY, Z, cmap='coolwarm', alpha=0.6, edgecolor='k')

    # Plot the margin planes (Z = ±1)
    ax.contour(XX, YY, Z, zdir='z', offset=0, levels=[-1, 0, 1], colors='k', linestyles=['--', '-', '--'])

    # Highlight support vectors in the base plane
    ax.scatter(model.support_vectors_[:, 0],
               model.support_vectors_[:, 1],
               0, s=100, facecolors='none', edgecolors='k', linewidth=1.5)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Decision function (margin distance)')
    ax.set_title("3D SVM Hyperplane Projection")
    plt.tight_layout()
    plt.show()
