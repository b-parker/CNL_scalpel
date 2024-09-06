import numpy as np
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from src.utilities.surface_utils import get_thresholded_curv, get_thresholded_thickness, find_label_boundary, get_faces_from_vertices
from src.classes.subject import ScalpelSubject
from typing import List
from scipy.spatial.distance import cdist
from src.utilities.surface_utils import cluster_label_KMeans


def gyral_gap(subject: ScalpelSubject, label_1:str | List[str], label_2:str | List[str], method='adjacency', clustering='pca'):
    """ 
    Compute the gyral gap between two labels

    Parameters
    ----------
    label_1 : numpy array or list
        The first label indexes
    label_2 : numpy array or list
        The second label indexes
    method : str, optional
        The method to compute the gyral gap. The default is 'adjacency'. Alternatives is 'geodesic'
    clustering : str, optional
        The clustering method. The default is 'pca'. Alternatives is 'none'.
    """
    
    # Combina labels if label_1 or label_2 are a list

    if isinstance(label_1, list):
        label_1_ind = np.concatenate((subject.labels[label_name][0] for label_name in label_1))
    if isinstance(label_2, list):
        label_2_ind = np.concatenate((subject.labels[label_name][0] for label_name in label_2))


    if clustering == 'pca':    
        ## PCA on these labels
        # combine labels and standardize
        combined_label_ind = np.concatenate((label_1_ind, label_2_ind))
        scaler = StandardScaler()
        label_12_faces = get_faces_from_vertices(subject.faces, combined_label_ind)
        label_12_boundary = find_label_boundary(label_12_faces)
        label_12_boundary_RAS = subject.ras_coords[label_12_boundary]
        points_scaled = scaler.fit_transform(label_12_boundary_RAS)

        # PCA
        pca = PCA(n_components=2)  
        points_pca = pca.fit_transform(points_scaled)

        points_for_clustering = points_pca[:, 0].reshape(-1, 1)
        label1_points = points_for_clustering[np.isin(label_12_boundary, label_1_ind)]
        label2_points = points_for_clustering[np.isin(label_12_boundary, label_2_ind)]

        # Cluster
        Ag_clust_1 = AgglomerativeClustering(n_clusters=2).fit_predict(label1_points)
        Ag_clust_2 = AgglomerativeClustering(n_clusters=3).fit_predict(label2_points)
        clusters2_adjusted = Ag_clust_2 + 5
        Ag_clust_3 = AgglomerativeClustering(n_clusters=4).fit_predict(points_for_clustering)

        ## Merge clusters
        cluster_merged = []
        cluster_1_copy = np.copy(Ag_clust_1)
        cluster_2_copy = np.copy(clusters2_adjusted)

        for i, boundary_ind in enumerate(label_12_boundary):
            if np.isin(boundary_ind, label_1_ind):
                cluster_merged.append(cluster_1_copy[0])
                cluster_1_copy = cluster_1_copy[1:]
            elif np.isin(boundary_ind, label_2_ind):
                cluster_merged.append(cluster_2_copy[0])
                cluster_2_copy = cluster_2_copy[1:]

        cluster_merged = np.array(cluster_merged)
        
        # identify cluster of each label
        cluster_merged_label1 = np.where(label_12_boundary[np.isin(label_12_boundary, label_1_ind)])
        cluster_merged_label2 = np.where(label_12_boundary[np.isin(label_12_boundary, label_2_ind)])

        
        # Find closest clusters and separate into sets for each label
        cluster_1_inds = [0, 1]
        cluster_2_inds = [5, 6, 7]
        set_label1 = [label_12_boundary_RAS[cluster_merged == cluster_1_inds[i]] for i in range(len(cluster_1_inds))]
        set_label2 = [label_12_boundary_RAS[cluster_merged == cluster_2_inds[i]] for i in range(len(cluster_2_inds))]
                
        # Initialize minimum median distance to a large number
        min_median_distance = float('inf')
        closest_arrays = None

        # Calculate the median pairwise distance for each pair of arrays (one from each set)
        for i, array1 in enumerate(set_label1):
            for j, array2 in enumerate(set_label2):
                median_distance = cdist(array1, array2).median()
                if median_distance < min_median_distance:
                    min_median_distance = median_distance
                    closest_arrays = (i, j)

        # Output the result
        closest_set1_array = set_label1[closest_arrays[0]]
        closest_set2_array = set_label2[closest_arrays[1]]

        # Cluster gyrus
        cluster_kmeans = cluster_label_KMeans(subject.gyrus[0], subject.ras_coords, n_clusters=300)


        # mask clusters and get adjacent nodes
        mask1 = np.equal(cluster_merged, cluster_1_inds[closest_arrays[0]])
        mask2 = np.equal(cluster_merged, cluster_2_inds[closest_arrays[1]])
        label_1_cluster_boundary = label_12_boundary[mask1]
        label_2_cluster_boundary = label_12_boundary[mask2]

        label_1_cluster_boundary_faces = get_faces_from_vertices(subject.faces, label_1_cluster_boundary, include_all=False)
        label_1_cluster_boundary_neighbors = np.unique(label_1_cluster_boundary_faces)

        label_2_cluster_boundary_faces = get_faces_from_vertices(subject.faces, label_2_cluster_boundary, include_all=False)
        label_2_cluster_boundary_neighbors = np.unique(label_2_cluster_boundary_faces)

        if method == 'adjacency':
            label_1_cluster_boundary_faces = get_faces_from_vertices(subject.faces, label_1_cluster_boundary, include_all=False)
            label_1_cluster_boundary_neighbors = np.unique(label_1_cluster_boundary_faces)

            label_2_cluster_boundary_faces = get_faces_from_vertices(subject.faces, label_2_cluster_boundary, include_all=False)
            label_2_cluster_boundary_neighbors = np.unique(label_2_cluster_boundary_faces)






