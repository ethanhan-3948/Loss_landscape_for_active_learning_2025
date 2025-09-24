import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from util.plot import plot_loss_landscape
from sklearn.decomposition import PCA

def select_optimal_clusters(X, k_min=2, k_max=None, random_state=0):
    """
    Choose the best number of clusters for KMeans using silhouette score.
    
    Args:
        X            : np.ndarray of shape (M, features)
        k_min        : int, minimum number of clusters to try (>=2)
        k_max        : int or None, maximum number of clusters to try;
                       if None, uses min(10, M-1)
        random_state : int, random seed for reproducibility
    
    Returns:
        best_k       : int, the number of clusters with highest silhouette score
    """
    M = X.shape[0]
    if k_max is None:
        k_max = min(10, M - 1)
    best_score = -1
    best_k = k_min
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state).fit(X)
        labels = km.labels_
        # silhouette_score requires at least 2 clusters and fewer than M clusters
        if 1 < len(np.unique(labels)) < M:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k = score, k
    return best_k
def approximate_subset_auto_k(samples,
                              n_clusters=None,
                              mae_threshold=1e-3,
                              max_iters=1000,
                              random_state=0,
                              distance_metric='norm'):
    """
    Find a small subset whose average loss landscape approximates the full average
    within a given MAE tolerance, automatically choosing cluster count if desired.
    
    Args:
        samples        : np.ndarray of shape (M, n, n, 1)
        n_clusters     : int or None; if None, automatically selected via silhouette
        mae_threshold  : float, stopping criterion on mean absolute error
        max_iters      : int, maximum number of refinement iterations
        random_state   : int, seed for clustering
        distance_metric: str, either 'norm' or 'cosine' for centroid distance calculation
    
    Returns:
        selected_idxs  : set of indices into `samples` forming the chosen subset
        L_full         : np.ndarray, the true full-list average after log transform
        L_approx       : np.ndarray, the final subset-average approximation after log transform
        mae_val        : float, final MAE between L_approx and L_full
        used_k         : int, the number of clusters actually used
    """
    # 1. Compute the full-list average (average raw then log)
    L_full = np.log(samples.mean(axis=0))            # (n, n, 1)
    f_full = L_full.ravel()
    
    # 2. Flatten for clustering
    M, n, _, _ = samples.shape
    X = samples.reshape(M, -1)               # (M, n*n)
    
    # 3. Decide number of clusters if not provided
    if n_clusters is None:
        n_clusters = select_optimal_clusters(X, random_state=random_state)
        print(f"Automatically selected n_clusters = {n_clusters}")
    
    # 4. Cluster into n_clusters groups
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    labels = kmeans.labels_
    
    # 5. Pick one representative per cluster (closest to centroid)
    selected_idxs = set()
    for k in range(n_clusters):
        members = np.where(labels == k)[0]
        centroid = kmeans.cluster_centers_[k]
        
        if distance_metric == 'norm':
            dists = np.linalg.norm(X[members] - centroid, axis=1)
            selected_idxs.add(members[np.argmin(dists)])
        elif distance_metric == 'cosine':
            # Compute cosine similarity (higher is better)
            similarities = np.sum(X[members] * centroid, axis=1) / (
                np.linalg.norm(X[members], axis=1) * np.linalg.norm(centroid)
            )
            selected_idxs.add(members[np.argmax(similarities)])
        else:
            raise ValueError("distance_metric must be either 'norm' or 'cosine'")
    
    # 6. Compute initial approximation and MAE (average raw then log)
    L_approx = np.log(samples[list(selected_idxs)].mean(axis=0))
    f_approx = L_approx.ravel()
    mae_val = np.mean(np.abs(f_full - f_approx))
    
    # 7. Iterative refinement to reduce MAE
    iters = 0
    while mae_val > mae_threshold and iters < max_iters:
        best_reduction = 0.0
        best_candidate = None
        
        for k in range(n_clusters):
            cluster_members = set(np.where(labels == k)[0])
            candidates = cluster_members - selected_idxs
            for c in candidates:
                trial_idxs = list(selected_idxs) + [c]
                # Average raw values then apply log
                L_trial = np.log(samples[trial_idxs].mean(axis=0)).ravel()
                trial_mae = np.mean(np.abs(f_full - L_trial))
                reduction = mae_val - trial_mae
                if reduction > best_reduction:
                    best_reduction, best_candidate = reduction, c
        
        if best_candidate is None:
            #print("No further improvement possible; stopping.")
            break
        
        selected_idxs.add(best_candidate)
        # Average raw values then apply log for final approximation
        L_approx = np.log(samples[list(selected_idxs)].mean(axis=0))
        f_approx = L_approx.ravel()
        mae_val = np.mean(np.abs(f_full - f_approx))
        iters += 1
    
    return selected_idxs, L_full, L_approx, mae_val, n_clusters
