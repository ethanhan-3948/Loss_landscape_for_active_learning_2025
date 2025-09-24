import matplotlib.pyplot as plt
import numpy as np
from util.general import flatten_and_vstack
import umap
from matplotlib.lines import Line2D


def plot_twin_umap_scatter(umap_data, n_neighbors=20, min_dist=0.05, labels1=None, labels2=None, label_name1=None, label_name2=None, label_type='categorical', landscape_array=None):
    """
    Generates a side-by-side scatter plot of UMAP embeddings with optional color coding based on two sets of labels.

    Parameters:
    - umap_data: numpy array, the UMAP transformed data
    - n_neighbors: int, number of neighbors used in UMAP
    - min_dist: float, minimum distance used in UMAP 
    - labels1: array-like, optional, the first set of labels for color coding the scatter plot
    - labels2: array-like, optional, the second set of labels for color coding the scatter plot
    - label_name1: str, optional, the name of the first label for the legend or color scale
    - label_name2: str, optional, the name of the second label for the legend or color scale
    - label_type: str, 'categorical', 'continuous', or 'boolean', specifies the type of labels
    """
    def process_labels(labels):
        unique_labels = None
        if labels is not None:
            if label_type == 'categorical' and isinstance(labels[0], str):
                unique_labels, labels = np.unique(labels, return_inverse=True)
            elif label_type == 'boolean' and isinstance(labels[0], bool):
                labels = labels.astype(int)
                unique_labels = ['False', 'True']
        return labels, unique_labels

    labels1, unique_labels1 = process_labels(labels1)
    labels2, unique_labels2 = process_labels(labels2)

    # Create the side-by-side scatter plots
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax, labels, label_name, unique_labels in zip(axs, [labels1, labels2], [label_name1, label_name2], [unique_labels1, unique_labels2]):
        if labels is not None:
            if label_type == 'continuous':
                scatter = ax.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, s=20, alpha=0.7, cmap='turbo')
                cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                if label_name:
                    cbar.set_label(label_name)
            else:
                scatter = ax.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, s=20, alpha=0.7, cmap='tab10')
                if unique_labels is not None:
                    if label_type == 'categorical':
                        handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(unique_labels[i]),
                                            markerfacecolor=plt.cm.tab10(i / len(unique_labels)), markersize=10)
                                 for i in range(len(unique_labels))]
                    else:  # boolean
                        handles = [plt.Line2D([0], [0], marker='o', color='w', label=unique_labels[i],
                                            markerfacecolor=plt.cm.tab10(i), markersize=10)
                                 for i in range(2)]
                    ax.legend(handles=handles, title=label_name, loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.scatter(umap_data[:, 0], umap_data[:, 1], s=20, alpha=0.7)

        ax.set_title(f'UMAP Projection of Loss Landscape:\n {landscape_array.name}\n' +
                 f'Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}',
                 fontsize=10, pad=10)
        ax.set_xlabel('UMAP Dimension 1', fontsize=10)
        ax.set_ylabel('UMAP Dimension 2', fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_umap_scatter(umap_data, n_neighbors=20, min_dist=0.05,
                      labels=None, label_name=None,
                      label_type='categorical', landscape_array=None):

    # --- encode categorical labels numerically ---
    if labels is not None and label_type == 'categorical':
        unique_labels, labels = np.unique(labels, return_inverse=True)
    else:
        unique_labels = None

    # --- begin plotting ---
    plt.figure(figsize=(11, 8))

    if labels is not None and label_type == 'categorical':
        # build discrete color list
        color_list = plt.cm.tab20.colors  # 10 distinct colors
        point_colors = [color_list[i % 10] for i in labels]

        plt.scatter(umap_data[:,0], umap_data[:,1],
                    c=point_colors, s=20, alpha=0.7)

        # build legend
        handles = []
        for i, ul in enumerate(unique_labels):
            handles.append(
                Line2D([0],[0], linestyle='None',
                       marker='o', markersize=8,
                       markerfacecolor=color_list[i % 10])
            )
        plt.legend(handles, unique_labels,
                   title=label_name,
                   loc='center left', bbox_to_anchor=(1, 0.5))

    elif labels is not None and label_type == 'continuous':
        scatter = plt.scatter(umap_data[:,0], umap_data[:,1],
                              c=labels, s=20, alpha=0.7,
                              cmap='turbo')
        cbar = plt.colorbar(scatter,
                            orientation='vertical',
                            fraction=0.046, pad=0.04)
        if label_name:
            cbar.set_label(label_name)

    else:
        plt.scatter(umap_data[:,0], umap_data[:,1],
                    s=20, alpha=0.7)

    plt.title(
        f'UMAP Projection of Loss Landscape:\n {getattr(landscape_array, "name", "")}\n'
        f'Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}',
        fontsize=14, pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.tick_params(labelsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.show()


def plot_umap_parameter_grid(landscape_array, n_neighbors_list, min_dist_list):
    """
    Creates a grid of UMAP plots showing embeddings with different hyperparameter combinations.
    
    Args:
        landscape_array: Array containing the data to embed
        n_neighbors_list: List of n_neighbors values to try
        min_dist_list: List of min_dist values to try
    """
    name = landscape_array.name
    landscape_array = flatten_and_vstack(landscape_array)   
    # Calculate number of subplots needed
    n_rows = len(n_neighbors_list)
    n_cols = len(min_dist_list)

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    fig.suptitle(f'UMAP for {name} with Different Hyperparameters', fontsize=16, y=1)

    # Iterate through parameter combinations
    for i, n_neighbors in enumerate(n_neighbors_list):
        for j, min_dist in enumerate(min_dist_list):
            # Initialize UMAP with current parameters
            umap_reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=2,
            )
            
            # Fit and transform the data
            embedding = umap_reducer.fit_transform(flatten_and_vstack(landscape_array))
            
            # Create scatter plot in corresponding subplot
            axes[i,j].scatter(
                embedding[:, 0],
                embedding[:, 1],
                s=5,
                alpha=0.5
            )
            
            axes[i,j].grid(True, linestyle='--', alpha=0.3)
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            
            # Only set xlabel for bottom row
            if i == n_rows-1:
                axes[i,j].set_xlabel(f'min_dist={min_dist}')
            
            # Only set ylabel for leftmost column
            if j == 0:
                axes[i,j].set_ylabel(f'n_neighbors={n_neighbors}')

    plt.tight_layout()
    plt.show()

def plot_loss_landscape(array, title, color_scale='turbo'):
    """
    Plots a non-interpolated image of a given (n, n, 1) shaped numpy array.

    Parameters:
    array (np.ndarray): The array to plot, with shape (n, n, 1).
    title (str): The title of the plot.
    color_scale (str, optional): The color scale to use for the plot. Defaults to None.
    """

    # Remove the last dimension to get a 2D array
    array_2d = array[:, :, 0]

    # Plot the image without interpolation
    plt.figure(figsize=(8, 6))
    plt.imshow(array_2d, cmap=color_scale, origin='lower', extent=(-array_2d.shape[0]/2 -0.5, array_2d.shape[0]/2 -0.5, -array_2d.shape[1]/2 -0.5, array_2d.shape[1]/2 -0.5))
    plt.colorbar()
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def visualize_image_clusters(arrays, cluster_labels, arrays_per_row=10, max_arrays_per_cluster=None):
    """
    Visualizes clusters of 3D arrays as 2D heatmaps.

    Parameters:
    arrays (list of np.ndarray): A list of 3D numpy arrays, each with shape (n, n, 1).
    cluster_labels (np.ndarray): An array of cluster labels corresponding to each array.
    arrays_per_row (int, optional): Number of arrays to display per row. Defaults to 10.
    max_arrays_per_cluster (int, optional): Maximum number of arrays to display per cluster. 
                                            If None, all arrays in the cluster are displayed.

    Returns:
    None
    """
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        cluster_indices = np.where(cluster_labels == label)[0]
        selected_indices = cluster_indices[:max_arrays_per_cluster]
        cluster_arrays = [arrays[i][:, :, 0] for i in selected_indices]  # Convert 3D arrays to 2D
        
        n_arrays = len(cluster_arrays)
        n_rows = (n_arrays + arrays_per_row - 1) // arrays_per_row
        
        plt.figure(figsize=(arrays_per_row * 1.5, n_rows * 1.5))
        for i, arr in enumerate(cluster_arrays):
            plt.subplot(n_rows, arrays_per_row, i + 1)
            plt.imshow(arr, cmap='viridis', origin='lower') 
            plt.axis('off')
        plt.suptitle(f'Cluster {label}', fontsize=16)
        plt.tight_layout()
        plt.show()

def plot_categorical_data(data):
        print(f"{data.dtype} data detected")
        plt.figure(figsize=(6,4))
        counts = data.value_counts()
        plt.bar(range(len(counts)), counts.values)
        plt.xticks(range(len(counts)), counts.index, rotation=45)
        plt.title(f"Bar Plot of {data.name}")
        plt.xlabel("Value")
        plt.ylabel("Count") 
        plt.tight_layout()
        plt.show()

def plot_numerical_data(data, bin_number=10, display_stats=True):
        """
        Analyze and visualize numerical data with binning.
        
        Args:
            data: pandas Series containing numerical data
            bin_number: int, number of bins for histogram (default 10)
            display_stats: bool, whether to display basic statistics (default True)
        """
        print(f"Numerical data detected for {data.name}")
        
        if display_stats:
            print("\nBasic Statistics:")
            print(f"Mean: {data.mean():.3f}")
            print(f"Median: {data.median():.3f}") 
            print(f"Std Dev: {data.std():.3f}")
            print(f"Min: {data.min():.3f}")
            print(f"Max: {data.max():.3f}")
        
        # Create bins and get bin assignments
        counts, bins = np.histogram(data, bins=bin_number)
        binned_labels = np.digitize(data, bins) - 1

        # Calculate mean and std for each bin
        bin_means = []
        bin_stds = []
        for i in range(bin_number):
            bin_samples = data[binned_labels == i]
            bin_means.append(np.mean(bin_samples))
            bin_stds.append(np.std(bin_samples))

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

        # Left subplot - histogram
        ax1.hist(data, bins=bin_number)
        ax1.set_title(f"Distribution of {data.name}")
        ax1.set_ylabel("Count")
        ax1.set_xlabel(data.name)

        # Right subplot - scatter with error bars
        ax2.errorbar(bin_means, bin_means, yerr=bin_stds, fmt='o', capsize=5)
        ax2.set_title(f"Bin Statistics for {data.name}")
        ax2.set_xlabel(f"{data.name} Bin Mean")
        ax2.set_ylabel(f"{data.name} Bin Mean ± Std")

        plt.tight_layout()
        plt.show()

        return binned_labels

