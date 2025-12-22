import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

# -----------------------------------------------------
# 1️⃣  Multi-Label SMOTE (core function)
# -----------------------------------------------------
def MLSMOTE(X, y, n_neighbors=5, target_ratio=1.0):
    """
    Multi-label SMOTE implementation with auto balancing.
    
    Parameters:
        X : np.ndarray or pd.DataFrame, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples, n_labels)
            Binary multi-label indicator matrix.
        n_neighbors : int
            Number of neighbors for interpolation.
        target_ratio : float
            Ratio of minority label frequency to match majority labels.
            (e.g. 1.0 means fully balanced)
    
    Returns:
        X_res, y_res : np.ndarray
            Augmented dataset after MLSMOTE.
    """
    X = np.array(X)
    y = np.array(y)
    n_labels = y.shape[1]
    
    # Compute label frequencies
    label_counts = y.sum(axis=0)
    max_count = label_counts.max()
    target_counts = (max_count * target_ratio).astype(int)
    
    X_res, y_res = [X.copy()], [y.copy()]
    
    # Perform MLSMOTE for each minority label
    for label_idx in range(n_labels):
        count = label_counts[label_idx]
        if count < target_counts:
            needed = target_counts - count
            minority_idx = np.where(y[:, label_idx] == 1)[0]
            
            if len(minority_idx) < 2:
                continue  # skip labels with too few samples
            
            # Fit nearest neighbors on minority samples
            nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(minority_idx)))
            nn.fit(X[minority_idx])
            
            # Generate synthetic samples
            synthetic_X, synthetic_y = [], []
            for _ in range(needed):
                idx = np.random.choice(minority_idx)
                neighbor_idx = np.random.choice(nn.kneighbors(X[idx].reshape(1, -1), return_distance=False)[0])
                
                lam = np.random.rand()
                new_x = X[idx] + lam * (X[neighbor_idx] - X[idx])
                
                # Combine label sets (union)
                new_y = np.maximum(y[idx], y[neighbor_idx])
                
                synthetic_X.append(new_x)
                synthetic_y.append(new_y)
            
            X_res.append(np.array(synthetic_X))
            y_res.append(np.array(synthetic_y))
    
    X_res = np.vstack(X_res)
    y_res = np.vstack(y_res)
    return X_res, y_res

# -----------------------------------------------------
# 2️⃣  Example usage on synthetic multi-label dataset
# -----------------------------------------------------
if __name__ == "__main__":
    # Suppose we have 6 samples, 3 labels
    X = np.random.rand(6, 4)
    y = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0],
    ])

    print("Original label counts:", y.sum(axis=0))
    X_res, y_res = MLSMOTE(X, y, n_neighbors=3, target_ratio=1.0)
    print("Resampled label counts:", y_res.sum(axis=0))
    print("X_res shape:", X_res.shape)

