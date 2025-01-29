from functools import cached_property
from typing import List, Tuple, Union, Callable
import numpy as np
import nibabel as nb
import os
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.spatial.distance import cdist
from src.utilities import freesurfer_utils as fsu
from src.utilities import surface_utils
from src.utilities import geometry_utils
from src.utilities.plotting import initialize_scene, plot, plot_label, remove_label, show_scene

class ScalpelSubject(object):
    def __init__(self, name, hemi, subjects_dir, surface_type="inflated"):
        self._name = name
        self._hemi = hemi
        self._surface_type = surface_type
        self._labels = defaultdict(list)
        self._subject_fs_path = Path(f'{subjects_dir}/{name}/')
        self._scene = None
        self._mesh = {}
    
        surface = nb.freesurfer.read_geometry(f'{subjects_dir}/{name}/surf/{hemi}.{surface_type}')
        self._curv = nb.freesurfer.read_morph_data(f'{subjects_dir}/{self._name}/surf/{self._hemi}.curv')
        self._ras_coords, self._faces = surface[0], surface[1]
        self._gyrus = fsu.get_gyrus(np.unique(self._faces), self._ras_coords, self._curv)
        self._sulcus = fsu.get_sulcus(np.unique(self._faces), self._ras_coords, self._curv)

    ############################
    # Properties
    ############################    
    @property
    def name(self):
        ## Subject ID
        return self._name

    @property
    def hemi(self):
        ## Hemisphere
        return self._hemi

    @property
    def surface_type(self):
        ## Surface Type - inflated, pial, white
        return self._surface_type

    @property
    def ras_coords(self):
        ## Right, Anterior, Superior Coordinates of all vertices
        return self._ras_coords

    @property
    def faces(self):
        ## Faces of the mesh
        return self._faces

    @property
    def vertex_indexes(self):
        ## Unique vertex indexes
        return np.unique(self.faces)

    @property
    def subject_fs_path(self):
        ## Path to subject's freesurfer directory
        return self._subject_fs_path

    @cached_property
    def mesh(self):
        gyrus_gray = [250, 250, 250]
        sulcus_gray = [130, 130, 130]
        if self.surface_type == 'inflated':
            print('Building cortical mesh (this may take 1 - 2 minutes)')
            gyrus_mesh = geometry_utils.make_mesh(self._ras_coords, self._faces, self._gyrus[0], face_colors=gyrus_gray)
            sulcus_mesh = geometry_utils.make_mesh(self._ras_coords, self._faces, self._sulcus[0], face_colors=sulcus_gray)
            self._mesh['gyrus'] = gyrus_mesh
            self._mesh['sulcus'] = sulcus_mesh
        else:
            self._mesh['cortex'] = geometry_utils.make_mesh(self._ras_coords, self._faces, self.vertex_indexes, face_colors=gyrus_gray)
        return self._mesh

    @property
    def curv(self):
        return self._curv

    @cached_property
    def thickness(self):
        return nb.freesurfer.read_morph_data(f'{self.subject_fs_path}/surf/{self.hemi}.thickness')

    @cached_property
    def gyrus(self):
        return self._gyrus

    @cached_property
    def sulcus(self):
        return self._sulcus

    @property
    def labels(self):
        """
        Access loaded labels through dictionary
    
        sub.labels['label_name'][0] # Vertex indexes
        sub.labels['label_name'][1] # RAS coordinates
        """
        return self._labels

    def load_label(self, label_name, label_idxs=None, label_RAS=None, custom_label_path=None):
        """
        Load a label into the subject class. Either loads from file or from input parameters.

        Parameters:
        - label_name (str): Name of the label.
        - label_idxs (np.array, optional): Vertex indices of the label. Defaults to None.
        - label_RAS (np.array, optional): RAS coordinates of the label. Defaults to None.

        Returns:
        - None 
        """
        if label_idxs is None or label_RAS is None:
            if custom_label_path is None:
                label_idxs, label_RAS = fsu.read_label(f'{self.subject_fs_path}/label/{self.hemi}.{label_name}.label')
    
            else:
                if isinstance(custom_label_path, str):
                    custom_label_path = Path(custom_label_path)
                label_idxs, label_RAS = fsu.read_label(custom_label_path / f'{self.hemi}.{label_name}.label')
        
        self._labels[label_name] = {
            'idxs': label_idxs,
            'RAS': label_RAS
        }

    def write_label(self, label_name, label_idxs = None, label_RAS = None, surface_type = "inflated", overwrite = False, custom_label_path = None):
        """
        Write a label to a file.

        Parameters:
        - label_name (str): Name of the label.
        - label_idxs (np.array): Vertex indices of the label.
        - label_RAS (np.array): RAS coordinates of the label.

        Returns:
        - None
        """
        if label_idxs is None or label_RAS is None:
            label_idxs = self.labels[label_name]['idxs']
            label_RAS = self.labels[label_name]['RAS']
        

        if custom_label_path is None:
            fsu.write_label(label_name, label_idxs, label_RAS, self.hemi, self.subject_fs_path.stem, self.surface_type, overwrite = overwrite)
        else:
            fsu.write_label(label_name, label_idxs, label_RAS, self.hemi, self.subject_fs_path.stem, self.surface_type, overwrite, custom_label_dir = custom_label_path)

    ############################
    # Visualization Methods
    ############################

    def plot(self, view='lateral', labels: List[str] = None):
        if self._mesh == {}:
            self.mesh
        if self._scene is None:
            self._scene = initialize_scene(self._mesh, view, self._hemi, self._surface_type)
        return plot(self._scene, view, labels, self.plot_label)

    def plot_label(self, label_name: str, view='lateral', label_ind=None, face_colors=None):
        if self._scene is None:
            initialize_scene()
        return plot_label(self._scene, self._ras_coords, self._faces, self._labels, label_name, view, self._hemi, face_colors, label_ind)

    def remove_label(self, label_name: str):
        return remove_label(self._scene, label_name)

    def show(self):
        return show_scene(self._scene)

    def create_subject(name, hemi="lh", surface_type="inflated"):
        subject = ScalpelSubject(name, hemi, surface_type)
        return subject
    

    ############################
    # Gyral Analysis Methods
    ############################

    def combine_labels(self, label_names: List[str], new_label_name: str) -> None:
        """
        Combine multiple labels into a single new label.

        Args:
            label_names (List[str]): List of label names to combine.
            new_label_name (str): Name for the new combined label.

        Raises:
            ValueError: If any of the input labels don't exist.
        """
        if not all(name in self.labels for name in label_names):
            raise ValueError("All input labels must exist in the subject.")

        combined_ind = np.unique(np.concatenate([self.labels[name][0] for name in label_names])).astype(int)
        combined_ras = self.ras_coords[combined_ind]
        self.load_label(new_label_name, combined_ind, combined_ras)

    def perform_boundary_analysis(self, label_name: str, method: str = 'pca', 
                                  n_components: int = 2, n_clusters: Union[int, List[int]] = [2, 3],
                                  clustering_algorithm: str = 'agglomerative') -> dict:
        """
        Perform boundary analysis on a label using PCA or direct clustering.

        Args:
            label_name (str): Name of the label to analyze.
            method (str): Analysis method ('pca' or 'direct').
            n_components (int): Number of PCA components (ignored if method is 'direct').
            n_clusters (Union[int, List[int]]): Number of clusters for each part of the label.
            clustering_algorithm (str): Clustering algorithm to use ('agglomerative', 'kmeans', or 'dbscan').

        Returns:
            dict: Analysis results including cluster assignments.

        Raises:
            ValueError: If the label doesn't exist or parameters are invalid.
        """
        if label_name not in self.labels:
            raise ValueError(f"Label '{label_name}' does not exist.")

        label_faces = geometry_utils.get_faces_from_vertices(self.faces, self.labels[label_name][0])
        label_boundary = surface_utils.find_label_boundary(label_faces)
        boundary_ras = self.ras_coords[label_boundary]

        if method == 'pca':
            scaler = StandardScaler()
            points_scaled = scaler.fit_transform(boundary_ras)
            pca = PCA(n_components=n_components)
            points_pca = pca.fit_transform(points_scaled)
            points_for_clustering = points_pca
        elif method == 'direct':
            points_for_clustering = boundary_ras
        else:
            raise ValueError("Method must be either 'pca' or 'direct'.")

        if isinstance(n_clusters, int):
            n_clusters = [n_clusters]

        cluster_results = []
        for n in n_clusters:
            if clustering_algorithm == 'agglomerative':
                clusters = AgglomerativeClustering(n_clusters=n).fit_predict(points_for_clustering)
            elif clustering_algorithm == 'kmeans':
                clusters = KMeans(n_clusters=n, random_state=42).fit_predict(points_for_clustering)
            elif clustering_algorithm == 'dbscan':
                clusters = DBSCAN(eps=0.5, min_samples=5).fit_predict(points_for_clustering)
                # Note: DBSCAN doesn't use n_clusters, it determines the number of clusters automatically
            else:
                raise ValueError("Unsupported clustering algorithm. Choose 'agglomerative', 'kmeans', or 'dbscan'.")
            cluster_results.append(clusters)

        return {
            'boundary': label_boundary,
            'boundary_ras': boundary_ras,
            'cluster_results': cluster_results
        }

    def find_closest_clusters(self, analysis_results1: dict, analysis_results2: dict) -> Tuple[int, int, float]:
        """
        Find the closest clusters between two label analysis results.

        Args:
            analysis_results1 (dict): Analysis results for the first label.
            analysis_results2 (dict): Analysis results for the second label.

        Returns:
            Tuple[int, int, float]: Indices of closest clusters and their median distance.
        """
        def median_pairwise_distance(array1, array2):
            distances = cdist(array1, array2, 'euclidean')
            return np.median(distances)

        min_median_distance = float('inf')
        closest_arrays = None

        for i, clusters1 in enumerate(analysis_results1['cluster_results']):
            for j, clusters2 in enumerate(analysis_results2['cluster_results']):
                for cluster1 in np.unique(clusters1):
                    for cluster2 in np.unique(clusters2):
                        array1 = analysis_results1['boundary_ras'][clusters1 == cluster1]
                        array2 = analysis_results2['boundary_ras'][clusters2 == cluster2]
                        if len(array1) > 0 and len(array2) > 0:  # Ensure non-empty arrays
                            median_distance = median_pairwise_distance(array1, array2)
                            if median_distance < min_median_distance:
                                min_median_distance = median_distance
                                closest_arrays = ((i, cluster1), (j, cluster2))

        if closest_arrays is None:
            raise ValueError("No valid cluster pairs found for comparison.")

        return closest_arrays[0], closest_arrays[1], min_median_distance

    def perform_gyral_clustering(self, n_clusters: int = 300, algorithm: str = 'kmeans') -> np.ndarray:
        """
        Perform clustering on gyral regions.

        Args:
            n_clusters (int): Number of clusters to create.
            algorithm (str): Clustering algorithm to use ('kmeans', 'agglomerative', or 'dbscan').

        Returns:
            np.ndarray: Cluster assignments for gyral vertices.

        Raises:
            ValueError: If an unsupported clustering algorithm is specified.
        """
        gyral_coords = self.ras_coords[self.gyrus[0]]

        if algorithm == 'kmeans':
            return KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(gyral_coords)
        elif algorithm == 'agglomerative':
            return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(gyral_coords)
        elif algorithm == 'dbscan':
            return DBSCAN(eps=0.5, min_samples=5).fit_predict(gyral_coords)
        else:
            raise ValueError("Unsupported clustering algorithm. Choose 'kmeans', 'agglomerative', or 'dbscan'.")

    @cached_property
    def gyral_clusters(self):
        """Perform K-means clustering on gyral regions."""
        return self.perform_gyral_clustering(n_clusters=300, algorithm='kmeans')

    def find_shared_gyral_clusters(self, label1: str, label2: str) -> np.ndarray:
        """
        Find shared gyral clusters between two labels.

        Args:
            label1 (str): Name of the first label.
            label2 (str): Name of the second label.

        Returns:
            np.ndarray: Indices of shared gyral clusters.

        Raises:
            ValueError: If either label doesn't exist.
        """
        if label1 not in self.labels or label2 not in self.labels:
            raise ValueError("Both labels must exist in the subject.")

        label1_faces = geometry_utils.get_faces_from_vertices(self.faces, self.labels[label1][0])
        label2_faces = geometry_utils.get_faces_from_vertices(self.faces, self.labels[label2][0])

        label1_neighbors = np.unique(label1_faces)
        label2_neighbors = np.unique(label2_faces)

        label1_gyral_neighbors = np.intersect1d(label1_neighbors, self.gyrus[0])
        label2_gyral_neighbors = np.intersect1d(label2_neighbors, self.gyrus[0])

        label1_gyral_clusters = np.unique(self.gyral_clusters[np.isin(self.gyrus[0], label1_gyral_neighbors)])
        label2_gyral_clusters = np.unique(self.gyral_clusters[np.isin(self.gyrus[0], label2_gyral_neighbors)])

        shared_gyral_clusters = np.intersect1d(label1_gyral_clusters, label2_gyral_clusters)
        return shared_gyral_clusters

    def get_shared_gyral_region(self, shared_gyral_clusters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the shared gyral region based on shared gyral clusters.

        Args:
            shared_gyral_clusters (np.ndarray): Array of shared gyral cluster indices.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices and RAS coordinates of the shared gyral region.
        """
        shared_gyral_mask = np.isin(self.gyral_clusters, shared_gyral_clusters)
        shared_gyral_index = self.gyrus[0][shared_gyral_mask]
        shared_gyral_ras = self.ras_coords[shared_gyral_index]
        return shared_gyral_index, shared_gyral_ras

    def find_gyral_gap(self, label1: str, label2: str, method: str = 'pca', n_components: int = 2, 
                       n_clusters: Union[int, List[int]] = [2, 3], clustering_algorithm: str = 'agglomerative') -> dict:
        """
        Find the gyral gap between two labels.

        Args:
            label1 (str): Name of the first label.
            label2 (str): Name of the second label.
            method (str): Analysis method ('pca' or 'direct').
            n_components (int): Number of PCA components (ignored if method is 'direct').
            n_clusters (Union[int, List[int]]): Number of clusters for each part of the label.
            clustering_algorithm (str): Clustering algorithm to use for boundary analysis.

        Returns:
            dict: Complete analysis results including the gyral gap.

        Raises:
            ValueError: If labels don't exist or parameters are invalid.
        """
        # Validate inputs
        if label1 not in self.labels or label2 not in self.labels:
            raise ValueError("Both labels must exist in the subject.")

        # Perform boundary analysis
        analysis1 = self.perform_boundary_analysis(label1, method, n_components, n_clusters, clustering_algorithm)
        analysis2 = self.perform_boundary_analysis(label2, method, n_components, n_clusters, clustering_algorithm)

        # Find closest clusters
        closest1, closest2, min_distance = self.find_closest_clusters(analysis1, analysis2)

        # Find shared gyral clusters
        shared_clusters = self.find_shared_gyral_clusters(label1, label2)

        # Get shared gyral region
        shared_index, shared_ras = self.get_shared_gyral_region(shared_clusters)

        return {
            'label1_analysis': analysis1,
            'label2_analysis': analysis2,
            'closest_clusters': (closest1, closest2),
            'min_cluster_distance': min_distance,
            'shared_gyral_clusters': shared_clusters,
            'shared_gyral_index': shared_index,
            'shared_gyral_ras': shared_ras
        }
    
    def label_centroid(self, label_name: str, load = True, centroid_face = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the centroid of a label.

        Args:
            label_name (str): Name of the label.
            centroid_face (bool): If True, return the centroid faces assopciated with the centroid. Defaults to False.

        Returns:
            np.ndarray: Centroid coordinates.
        """

        # get the faces associate with the label
        label_faces_ind = np.where(np.isin(self.faces, self.labels[label_name]['idxs']))[0]
        label_faces = self.faces[label_faces_ind]

        # Calculate centroid 
        centroid_ras = surface_utils.calculate_geometric_centroid(self.ras_coords, label_faces)
        centroid_surface_vertex = surface_utils.find_closest_vertex(centroid_ras, self.ras_coords)[0]
        centroid_surface_ras = self.ras_coords[centroid_surface_vertex]

        # If getting the faces, return the faces and the centroid
        if centroid_face:
            centroid_faces = self.faces[np.where(np.isin(self.faces, centroid_surface_vertex))[0]]
            centroid_faces_vertices = np.array([np.unique(self.faces[centroid_faces])])
            centroid_ras = np.array([self.ras_coords[centroid_faces]])
            if load:
                self.load_label(f'{label_name}_centroid', label_idxs=centroid_faces_vertices, label_RAS=centroid_ras)
            return centroid_faces_vertices, centroid_ras
        else:
            centroid_surface_vertex = np.array([centroid_surface_vertex])
            centroid_surface_ras = np.array([centroid_surface_ras])
            if load:
                self.load_label(f'{label_name}_centroid', label_idxs=centroid_surface_vertex, label_RAS=centroid_surface_ras)
            return centroid_surface_vertex, centroid_surface_ras
        


