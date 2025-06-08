import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_svm_boundary(model, A, b):
    
    # Ensure inputs are numpy arrays
    X = np.asarray(A)
    y = np.asarray(b)

    # Scatter plot of the data
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='seismic')

    # Plot decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate the model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # To suppress warning about Feature names while plotting the graph
    if hasattr(A, "columns"):
        xy = pd.DataFrame(xy, columns=A.columns)
    
    Z = model.decision_function(xy).reshape(XX.shape)

    # Plot the decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
               alpha=0.5, linestyles=['--', '-', '--'])

    # Plot support vectors
    ax.scatter(model.support_vectors_[:, 0],
               model.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='k')

    plt.title("SVC Decision Boundary with Margins")
    plt.show()
