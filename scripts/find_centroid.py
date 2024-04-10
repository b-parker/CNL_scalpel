import numpy as np

def find_centroid(label_idx, label_RAS):
    """
    Find the centroid of a label

    INPUT:
    label_idx: np.array - array of indices of label
    label_RAS: np.array - array of RAS coordinates of label

    OUTPUT:
    centroid_idx: np.array - array idx of centroid of label
    centroid_RAS: np.array - RAS coordinates of the centroid
    """
    R_val = np.mean(label_RAS[:, 0])
    A_val = np.mean(label_RAS[:, 1])
    S_val = np.mean(label_RAS[:, 2])

    nearest_vertex_idx = np.argmin(np.linalg.norm(label_RAS - [R_val, A_val, S_val], axis=1))
    centroid_RAS = np.array([label_RAS[nearest_vertex_idx]])
    centroid_idx = np.array([label_idx[nearest_vertex_idx]])
    return centroid_idx, centroid_RAS
    