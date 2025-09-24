from sklearn.metrics import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def manual_spectral_clustering(data, affinity_type='rbf', affinity_param=1.0, visualize_affinity_scree=False, n_clusters=2, visualize_eigenvectors=False):
    """
    Perform manual spectral clustering on the given data.

    This function computes the affinity matrix, normalized Laplacian, and eigenvectors
    to transform the data into an embedding space. It then applies k-means clustering
    on this embedding space to obtain cluster labels.

    Parameters:
    data (np.ndarray): The input data for clustering.
    affinity_type (str): The type of affinity to use ('rbf' or 'knn').
    affinity_param (float or int): The parameter for the affinity function. 
                                   For 'rbf', it is sigma. For 'knn', it is the number of neighbors.
    visualize_affinity_scree (bool): Whether to visualize the affinity matrix and scree plot of eigenvalues.
    n_clusters (int): The number of clusters (and eigenvectors) to use.
    visualize_eigenvectors (bool): Whether to visualize the eigenvectors.

    Returns:
    np.ndarray: The cluster labels for the data.
    """
    # Step 1: Compute the affinity matrix
    if affinity_type == 'rbf':
        # Use RBF kernel to compute the affinity matrix
        affinity_matrix = pairwise_kernels(data, metric='rbf', gamma=1/(2*affinity_param**2))
    elif affinity_type == 'knn':
        # Use k-NN graph to compute the affinity matrix
        affinity_matrix = kneighbors_graph(data, n_neighbors=int(affinity_param), mode='connectivity', include_self=True).toarray()
    else:
        # Raise an error if the affinity type is invalid
        raise ValueError("Invalid affinity type. Choose 'rbf' or 'knn'.")

    # Step 2: Compute the normalized Laplacian matrix
    laplacian_matrix = np.eye(len(affinity_matrix)) - normalize(affinity_matrix, norm='l1', axis=1)

    # Step 3: Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = eigh(laplacian_matrix)

    # Visualize the affinity matrix and scree plot if requested
    if visualize_affinity_scree:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot the affinity matrix
        im = axs[0].imshow(affinity_matrix, cmap='viridis')
        axs[0].set_title('Affinity Matrix')
        fig.colorbar(im, ax=axs[0])
        
        # Plot the scree plot of eigenvalues
        axs[1].plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
        axs[1].set_title('Scree Plot of Eigenvalues')
        axs[1].set_xlabel('Index')
        axs[1].set_ylabel('Eigenvalue')
        axs[1].grid(True)
        
        plt.tight_layout()
        plt.show()

    # Step 4: Transform data into embedding space using the first n_clusters eigenvectors
    sorted_indices = np.argsort(eigenvalues)
    embedding = eigenvectors[:, sorted_indices[:n_clusters]]

    # Visualize the eigenvectors if requested
    if visualize_eigenvectors:
        num_plots = min(n_clusters, len(eigenvectors))
        n_rows = (num_plots + 3) // 4
        fig, axs = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
        axs = axs.flatten()
        for i in range(num_plots):
            axs[i].scatter(range(len(eigenvectors)), eigenvectors[:, sorted_indices[i]], alpha=0.7)
            axs[i].set_title(f'Eigenvector {i+1} (Eigenvalue: {eigenvalues[sorted_indices[i]]:.4f})')
            axs[i].set_xlabel('Index')
            axs[i].set_ylabel('Entry')
        for j in range(num_plots, len(axs)):
            fig.delaxes(axs[j])
        plt.tight_layout()
        plt.show()

    # Step 5: Perform k-means on the embedding space
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embedding)

    return labels
