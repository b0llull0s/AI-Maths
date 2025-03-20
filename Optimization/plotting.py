import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_optimization_path_2d(func, history, bounds, title="Optimization Path", 
                             resolution=100, contour_levels=20):
    """
    Plot the optimization path on a 2D contour plot.
    
    :param func: The objective function (takes a 2D array as input).
    :param history: History of positions visited during optimization.
    :param bounds: Bounds for x and y axes [(x_min, x_max), (y_min, y_max)].
    :param title: Plot title.
    :param resolution: Resolution of the contour plot.
    :param contour_levels: Number of contour levels.
    """
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, contour_levels, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='Function Value')
    
    plt.plot(history[:, 0], history[:, 1], 'o-', color='red', linewidth=1.5, markersize=3)
    plt.plot(history[0, 0], history[0, 1], 'o', color='blue', markersize=6, label='Start')
    plt.plot(history[-1, 0], history[-1, 1], 'o', color='green', markersize=6, label='End')
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

def plot_optimization_path_3d(func, history, bounds, title="Optimization Path in 3D", 
                             resolution=50, view_angle=(30, 45)):
    """
    Plot the optimization path on a 3D surface.
    
    :param func: The objective function (takes a 2D array as input).
    :param history: History of positions visited during optimization.
    :param bounds: Bounds for x and y axes [(x_min, x_max), (y_min, y_max)].
    :param title: Plot title.
    :param resolution: Resolution of the surface plot.
    :param view_angle: Viewing angle (elevation, azimuth).
    """
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=True)
    
    z_history = np.array([func(point) for point in history])
    ax.plot(history[:, 0], history[:, 1], z_history, 'o-', color='red', linewidth=2, markersize=4)
    ax.plot([history[0, 0]], [history[0, 1]], [z_history[0]], 'o', color='blue', markersize=8, label='Start')
    ax.plot([history[-1, 0]], [history[-1, 1]], [z_history[-1]], 'o', color='green', markersize=8, label='End')
    
    ax.view_init(view_angle[0], view_angle[1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(title)
    
    fig.colorbar(surface, shrink=0.5, aspect=5, label='Function Value')
    
    return fig

def plot_convergence(history_values, methods_names, log_scale=True, title="Convergence Comparison"):
    """
    Plot convergence of different optimization methods.
    
    :param history_values: List of arrays containing objective values for each method.
    :param methods_names: Names of the methods.
    :param log_scale: Whether to use log scale for y-axis.
    :param title: Plot title.
    """
    plt.figure(figsize=(10, 6))
    
    for i, history in enumerate(history_values):
        plt.plot(range(len(history)), history, '-', linewidth=2, label=methods_names[i])
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    
    if log_scale and all(np.all(h > 0) for h in history_values):
        plt.yscale('log')
    
    plt.grid(True)
    plt.legend()
    
    return plt.gcf()