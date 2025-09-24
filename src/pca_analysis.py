from sklearn.decomposition import PCA
from util.landscape_processing import restore_to_square_shape
from util.plot import plot_loss_landscape
import matplotlib.pyplot as plt
import numpy as np

def visualize_top_pcs(pca, n_top_pcs, top_features=15):
    """
    Analyzes the top principal components by displaying histograms of the top features
    and restoring the proportion array to its original shape for visualization.

    Parameters:
    pca (PCA): The PCA object containing components and other attributes.
    n_top_pcs (int): The number of top principal components to analyze.
    top_features (int): The number of top features to display in the histogram for each PC.
                       Default is 20.

    Returns:
    None
    """
    # Create a single figure with n_top_pcs rows and 2 columns
    fig, axs = plt.subplots(n_top_pcs, 2, figsize=(8, 4*n_top_pcs))
    
    for pc_index in range(n_top_pcs):
        pc_loadings = np.abs(pca.components_[pc_index])
        feature_proportions = pc_loadings / np.linalg.norm(pc_loadings)
        top_feature_indices = np.argsort(feature_proportions)[-top_features:]
        sorted_indices = np.argsort(-feature_proportions[top_feature_indices])
        top_features_proportions = feature_proportions[top_feature_indices][sorted_indices]
        top_features_names = [str(top_feature_indices[i]) for i in sorted_indices]

        # Plot the histogram
        ax1 = axs[pc_index, 0]
        ax1.bar(range(len(top_features_names)), top_features_proportions, color='skyblue')
        ax1.set_ylabel('Proportion Value')
        ax1.set_xlabel('Feature Index')
        ax1.set_title(f'Top {top_features} Feature Proportions for PC {pc_index + 1}')
        ax1.set_xticks(range(len(top_features_names)))
        ax1.set_xticklabels(top_features_names, rotation=60)
        range_ = max(top_features_proportions) - min(top_features_proportions)
        ax1.set_ylim(min(top_features_proportions) - 0.2*range_, 0.2 * range_ + max(top_features_proportions))

        # Plot the loss landscape
        restored_array = restore_to_square_shape(feature_proportions)
        ax2 = axs[pc_index, 1]
        im = ax2.imshow(restored_array[:, :, 0], cmap='viridis', origin='lower')
        ax2.set_title(f'Feature Proportions for PC {pc_index + 1}')
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')
        plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    plt.show()


def plot_pairwise_pc(pca_transformed_data, n_pcs_to_visualize, labels=None, label_name=None, label_type='categorical'):
    """
    Generates a subplot for pairwise scatter plots of the specified number of principal components.

    Parameters:
    - pca_transformed_data: numpy array, the PCA transformed data.
    - n_pcs_to_visualize: int, the number of principal components to visualize.
    - labels: array-like, optional, the labels for color coding the scatter plots.
    - label_name: str, optional, the name of the label for the legend or color scale.
    - label_type: str, 'categorical', 'continuous', or 'boolean', specifies the type of labels.

    Returns:
    - None, displays the scatter plots.
    """
    unique_labels = None
    # Convert categorical string labels to integers if necessary
    if labels is not None:
        if label_type == 'categorical' and isinstance(labels[0], str):
            unique_labels, labels = np.unique(labels, return_inverse=True)
        elif label_type == 'boolean' and isinstance(labels[0], bool):
            labels = labels.astype(int)
            unique_labels = ['False', 'True']

    # Determine the number of subplots needed
    fig, axs = plt.subplots(n_pcs_to_visualize, n_pcs_to_visualize, figsize=(8, 8))
    
    # Loop through each pair of principal components
    for i in range(n_pcs_to_visualize):
        for j in range(n_pcs_to_visualize):
            if i != j:
                if labels is not None:
                    if label_type == 'continuous':
                        scatter = axs[j, i].scatter(pca_transformed_data[:, i], pca_transformed_data[:, j], c=labels, alpha=0.5, cmap='turbo')
                    else:
                        scatter = axs[j, i].scatter(pca_transformed_data[:, i], pca_transformed_data[:, j], c=labels, alpha=0.5, cmap='tab10')
                else:
                    scatter = axs[j, i].scatter(pca_transformed_data[:, i], pca_transformed_data[:, j], alpha=0.5)
                if i == 0:
                    axs[j, i].set_ylabel(f'PC {j+1}')
                if j == n_pcs_to_visualize - 1:
                    axs[j, i].set_xlabel(f'PC {i+1}')
            else:
                axs[j, i].text(0.5, 0.5, f'PC {i+1}', fontsize=12, ha='center')
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])

    if labels is not None:
        if label_type == 'continuous':
            cbar = fig.colorbar(scatter, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
            if label_name:
                cbar.set_label(label_name)
        elif label_type == 'categorical' and unique_labels is not None:
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(unique_labels[i]), 
                                  markerfacecolor=plt.cm.tab10(i / len(unique_labels)), markersize=10) 
                       for i in range(len(unique_labels))]
            legend = axs[0, 0].legend(handles=handles, title=label_name, loc='upper right')
        elif label_type == 'boolean':
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=unique_labels[i], 
                                  markerfacecolor=plt.cm.tab10(i), markersize=10) 
                       for i in range(2)]
            legend = axs[0, 0].legend(handles=handles, title=label_name, loc='upper right')

    plt.show()

def plot_explained_variance(pca):
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), cumulative_explained_variance, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components')
    plt.grid(True)
    plt.show()

