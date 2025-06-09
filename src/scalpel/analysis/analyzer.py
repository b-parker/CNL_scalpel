from __future__ import annotations
from typing import List, Tuple, Dict, Union, TYPE_CHECKING, Optional
import numpy as np
import nibabel as nib
from functools import cached_property
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN

# Import utilities
from scalpel.utils import surface_utils
from scalpel.utils import freesurfer_utils as fsu

if TYPE_CHECKING:
    from scalpel.subject import ScalpelSubject

class ScalpelAnalyzer:
    """
    Class for analyzing brain surface data.
    
    This class provides analysis functionality for a ScalpelSubject,
    including surface analysis, gyral/sulcal analysis, and clustering.
    """
    
    def __init__(self, subject: 'ScalpelSubject'):
        """
        Initialize a ScalpelAnalyzer.
        
        Parameters:
        -----------
        subject : ScalpelSubject
            The subject to analyze
        """
        self._subject = subject
    
    @property
    def subject(self):
        """Get the associated ScalpelSubject."""
        return self._subject
    
    @cached_property
    def adjacency(self):
        """Adjacency matrix of the mesh"""
        return surface_utils.mesh_to_adjacency(
            self._subject.faces, 
            self._subject.vertex_indexes
        )
    
    @cached_property
    def curv(self):
        """Curvature values from FreeSurfer .curv file"""
        curv_path = self._subject.subject_fs_path / 'surf' / f'{self._subject.hemi}.curv'
        return nib.freesurfer.read_morph_data(str(curv_path))
    
    @cached_property
    def thickness(self):
        """Cortical thickness values from FreeSurfer .thickness file"""
        thickness_path = self._subject.subject_fs_path / 'surf' / f'{self._subject.hemi}.thickness'
        return nib.freesurfer.read_morph_data(str(thickness_path))
    
    @cached_property
    def sulc_vals(self):
        """Sulcal depth values from FreeSurfer .sulc file"""
        sulc_path = self._subject.subject_fs_path / 'surf' / f'{self._subject.hemi}.sulc'
        return nib.freesurfer.read_morph_data(str(sulc_path))
    
    @cached_property
    def pial_v(self):
        """Pial surface vertices (required for sulcal depth calculation)"""
        pial_path = self._subject.subject_fs_path / 'surf' / f'{self._subject.hemi}.pial'
        pial_verts, _ = nib.freesurfer.read_geometry(str(pial_path))
        return pial_verts
    
    @cached_property
    def gyrif_v(self):
        """Gyrus-inflated surface vertices (pial-outer-smoothed)"""
        gyrif_path = self._subject.subject_fs_path / 'surf' / f'{self._subject.hemi}.pial-outer-smoothed'
        gyrif_verts, _ = nib.freesurfer.read_geometry(str(gyrif_path))
        return gyrif_verts
    
    @cached_property
    def gyrus(self):
        """Get gyral vertices based on curvature"""
        return fsu.get_gyrus(
            np.unique(self._subject.faces), 
            self._subject.surface_RAS, 
            self.curv
        )
    
    @cached_property
    def sulcus(self):
        """Get sulcal vertices based on curvature"""
        return fsu.get_sulcus(
            np.unique(self._subject.faces), 
            self._subject.surface_RAS, 
            self.curv
        )
    
    @cached_property
    def gyral_clusters(self):
        """Perform K-means clustering on gyral regions."""
        return self.perform_gyral_clustering(n_clusters=300, algorithm='kmeans')
    
    def find_label_boundary(self, label_name : str):
        return surface_utils.find_label_boundary(self.subject.labels[label_name].faces)
        
    def find_shared_gyral_clusters(self, label1: str, label2: str) -> np.ndarray:
        """
        Find shared gyral clusters between two labels.

        Parameters:
            label1 (str): Name of the first label.
            label2 (str): Name of the second label.

        Returns:
            np.ndarray: Indices of shared gyral clusters.

        Raises:
            ValueError: If either label doesn't exist.
        """
        if label1 not in self._subject.labels or label2 not in self._subject.labels:
            raise ValueError("Both labels must exist in the subject.")

        label1_faces = surface_utils.get_faces_from_vertices(
            self._subject.faces, 
            self._subject.labels[label1].vertex_indexes, 
            include_all=False
        )
        label2_faces = surface_utils.get_faces_from_vertices(
            self._subject.faces, 
            self._subject.labels[label2].vertex_indexes, 
            include_all=False
        )

        label1_neighbors = np.unique(label1_faces)
        label2_neighbors = np.unique(label2_faces)

        label1_gyral_neighbors = np.intersect1d(label1_neighbors, self.gyrus[0])
        label2_gyral_neighbors = np.intersect1d(label2_neighbors, self.gyrus[0])

        label1_gyral_clusters = np.unique(self.gyral_clusters[np.isin(self.gyrus[0], label1_gyral_neighbors)])
        label2_gyral_clusters = np.unique(self.gyral_clusters[np.isin(self.gyrus[0], label2_gyral_neighbors)])

        shared_gyral_clusters = np.intersect1d(label1_gyral_clusters, label2_gyral_clusters)
        return shared_gyral_clusters
    
    def perform_gyral_clustering(self, n_clusters: int = 300, algorithm: str = 'kmeans') -> np.ndarray:
        """
        Perform clustering on gyral regions.

        Parameters:
            n_clusters (int): Number of clusters to create.
            algorithm (str): Clustering algorithm to use ('kmeans', 'agglomerative', or 'dbscan').

        Returns:
            np.ndarray: Cluster assignments for gyral vertices.

        Raises:
            ValueError: If an unsupported clustering algorithm is specified.
        """
        gyral_coords = self._subject.surface_RAS[self.gyrus[0]]

        if algorithm == 'kmeans':
            return KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(gyral_coords)
        elif algorithm == 'agglomerative':
            return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(gyral_coords)
        elif algorithm == 'dbscan':
            return DBSCAN(eps=0.5, min_samples=5).fit_predict(gyral_coords)
        else:
            raise ValueError("Unsupported clustering algorithm. Choose 'kmeans', 'agglomerative', or 'dbscan'.")
    
    def perform_boundary_analysis(self, label_name: str, method: str = 'pca', 
                                  n_components: int = 2, n_clusters: Union[int, List[int]] = [2, 3],
                                  clustering_algorithm: str = 'agglomerative') -> dict:
        """
        Perform boundary analysis on a label using PCA or direct clustering.

        Parameters:
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
        if label_name not in self._subject.labels:
            raise ValueError(f"Label '{label_name}' does not exist.")

        label_faces = surface_utils.get_faces_from_vertices(
            self._subject.faces, 
            self._subject.labels[label_name].vertex_indexes
        )
        label_boundary = surface_utils.find_label_boundary(label_faces)
        boundary_ras = self._subject.surface_RAS[label_boundary]

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

        Parameters:
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
    
    def get_shared_gyral_region(self, shared_gyral_clusters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the shared gyral region based on shared gyral clusters.

        Parameters:
            shared_gyral_clusters (np.ndarray): Array of shared gyral cluster indices.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices and RAS coordinates of the shared gyral region.
        """
        shared_gyral_mask = np.isin(self.gyral_clusters, shared_gyral_clusters)
        shared_gyral_index = self.gyrus[0][shared_gyral_mask]
        shared_gyral_ras = self._subject.surface_RAS[shared_gyral_index]
        return shared_gyral_index, shared_gyral_ras
    
    def find_gyral_gap(self, label1: str, label2: str, method: str = 'pca', n_components: int = 2, 
                       n_clusters: Union[int, List[int]] = [2, 3], clustering_algorithm: str = 'agglomerative',
                       disjoints: bool = True, load_label: bool = True) -> dict:
        """
        Find the gyral gap between two labels.

        Parameters:
            label1 (str): Name of the first label.
            label2 (str): Name of the second label.
            method (str): Analysis method ('pca' or 'direct').
            n_components (int): Number of PCA components (ignored if method is 'direct').
            n_clusters (Union[int, List[int]]): Number of clusters for each part of the label.
            clustering_algorithm (str): Clustering algorithm to use for boundary analysis.
            disjoints (bool): If True, find largest disjointed shared gyral region.

        Returns:
            dict: Complete analysis results including the gyral gap.

        Raises:
            ValueError: If labels don't exist or parameters are invalid.
        """
        if label1 not in self._subject.labels or label2 not in self._subject.labels:
            raise ValueError("Both labels must exist in the subject.")

        # Get boundary and cluster boundary vertices
        analysis1 = self.perform_boundary_analysis(label1, method, n_components, n_clusters, clustering_algorithm)
        analysis2 = self.perform_boundary_analysis(label2, method, n_components, n_clusters, clustering_algorithm)

        # Find closest clusters
        closest1, closest2, min_distance = self.find_closest_clusters(analysis1, analysis2)

        shared_clusters = self.find_shared_gyral_clusters(label1, label2)

        # Get shared gyral region
        shared_index, shared_ras = self.get_shared_gyral_region(shared_clusters)

        # If disjoints, find largest disjointed shared gyral region
        if disjoints and len(shared_index) > 0:
            shared_gyral_faces = surface_utils.get_faces_from_vertices(self._subject.faces, shared_index)
            disjoints = surface_utils.get_label_subsets(shared_gyral_faces, self._subject.faces)

            if disjoints:
                disjoints.sort(key=lambda x: len(x), reverse=True)
                shared_index = np.unique(disjoints[0])
                shared_ras = self._subject.surface_RAS[shared_index]

        if load_label == True:
            self._subject.load_label(
                f"{label1}_{label2}_shared_gyral", 
                label_idxs=shared_index, 
                label_RAS=shared_ras
            )

        return {
            'label1_analysis': analysis1,
            'label2_analysis': analysis2,
            'closest_clusters': (closest1, closest2),
            'min_cluster_distance': min_distance,
            'shared_gyral_clusters': shared_clusters,
            'shared_gyral_index': shared_index,
            'shared_gyral_ras': shared_ras
        }
        
    def analyze_sulcal_gyral_relationships(self, label_name: str, gyral_clusters: int = 300, 
                                           sulcal_clusters: Optional[int] = None, 
                                           algorithm: str = 'kmeans', 
                                           load_results: bool = True) -> dict:
        """
        Comprehensive analysis of sulcal-gyral relationships:
        1) Clusters the gyri
        2) Clusters the sulci
        3) For each sulcal cluster, finds adjacent gyral clusters
        4) Gets centroids of all clusters
        5) Determines anterior/posterior relationships
        
        Parameters:
            label_name (str): Name of the sulcal label
            gyral_clusters (int): Number of clusters for gyral clustering
            sulcal_clusters (int): Number of clusters for sulcal clustering
            algorithm (str): Clustering algorithm ('kmeans', 'agglomerative', or 'dbscan')
            load_results (bool): Whether to load result labels in the subject
            
        Returns:
            dict: Complete analysis results
        """
        if label_name not in self._subject.labels:
            raise ValueError(f"Sulcal label '{label_name}' not found in subject")
        
        if sulcal_clusters is None:
            sulcal_clusters = self._subject.labels[label_name].vertex_indexes.shape[0] // 400
            sulcal_clusters = max(sulcal_clusters, 2)  # Ensure at least 2 clusters
        
        results = {
            'sulcal_clusters': {},
            'gyral_clusters': {},
            'adjacency_map': {},
            'anterior_gyri': [],
            'posterior_gyri': []
        }
        
        # Cluster the gyri
        if not hasattr(self, 'gyral_clusters') or self.gyral_clusters is None:
            gyral_cluster_assignments = self.perform_gyral_clustering(
                n_clusters=gyral_clusters, algorithm=algorithm)
        else:
            gyral_cluster_assignments = self.gyral_clusters
        
        unique_gyral_clusters = np.unique(gyral_cluster_assignments)
        
        for cluster_id in unique_gyral_clusters:
            cluster_indices = np.where(gyral_cluster_assignments == cluster_id)[0]
            cluster_vertices = self.gyrus[0][cluster_indices]
            cluster_ras = self._subject.surface_RAS[cluster_vertices]
            
            results['gyral_clusters'][cluster_id] = {
                'vertices': cluster_vertices,
                'ras_coords': cluster_ras,
            }
        
        # Cluster the sulcus
        sulcus_vertices = self._subject.labels[label_name].vertex_indexes
        sulcus_ras = self._subject.surface_RAS[sulcus_vertices]
        
        if algorithm == 'kmeans':
            sulcal_clustering = KMeans(n_clusters=sulcal_clusters, random_state=42, n_init="auto")
        elif algorithm == 'agglomerative':
            sulcal_clustering = AgglomerativeClustering(n_clusters=sulcal_clusters)
        elif algorithm == 'dbscan':
            sulcal_clustering = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError("Unsupported clustering algorithm")
        
        sulcal_cluster_assignments = sulcal_clustering.fit_predict(sulcus_ras)
        unique_sulcal_clusters = np.unique(sulcal_cluster_assignments)
        
        for cluster_id in unique_sulcal_clusters:
            cluster_indices = np.where(sulcal_cluster_assignments == cluster_id)[0]
            cluster_vertices = sulcus_vertices[cluster_indices]
            cluster_ras = sulcus_ras[cluster_indices]
            centroid = np.mean(cluster_ras, axis=0)
            
            if load_results:
                cluster_label_name = f"{label_name}_cluster_{cluster_id}"
                self._subject.load_label(
                    cluster_label_name, 
                    label_idxs=cluster_vertices, 
                    label_RAS=cluster_ras
                )
            
            results['sulcal_clusters'][cluster_id] = {
                'vertices': cluster_vertices,
                'ras_coords': cluster_ras,
                'centroid': centroid
            }
        
        # For each sulcal cluster, find adjacent gyral clusters
        for sulcal_id in results['sulcal_clusters']:
            sulcal_cluster_vertices = results['sulcal_clusters'][sulcal_id]['vertices']
            
            sulcal_faces = surface_utils.get_faces_from_vertices(self._subject.faces, sulcal_cluster_vertices)
            sulcal_boundary = surface_utils.find_label_boundary(sulcal_faces)
            
            boundary_neighbors = set()
            for face in self._subject.faces:
                if any(v in sulcal_boundary for v in face):
                    boundary_neighbors.update(face)
            
            all_sulcal_vertices = set(sulcus_vertices)
            boundary_neighbors = boundary_neighbors - all_sulcal_vertices
            
            gyral_vertices = set(self.gyrus[0])
            adjacent_gyral_vertices = np.array(list(boundary_neighbors & gyral_vertices), dtype=int)
            
            adjacent_gyral_clusters = set()
            for v in adjacent_gyral_vertices:
                gyral_idx = np.where(self.gyrus[0] == v)[0]
                if len(gyral_idx) > 0:
                    cluster = gyral_cluster_assignments[gyral_idx[0]]
                    adjacent_gyral_clusters.add(cluster)
            
            results['adjacency_map'][sulcal_id] = list(adjacent_gyral_clusters)
            
            # Compare centroid RAS to determine anterior/posterior relationship
            sulcal_centroid = self.label_centroid(label_name, load=False, custom_vertexes=results['sulcal_clusters'][sulcal_id]['vertices'])
            
            for gyral_id in adjacent_gyral_clusters:
                gyral_centroid = self.label_centroid(label_name, load=False, custom_vertexes=results['gyral_clusters'][gyral_id]['vertices'])
                
                # In RAS coordinates, Y-axis corresponds to anterior-posterior direction
                # Higher Y values are more anterior
                is_anterior = gyral_centroid[1] > sulcal_centroid[1]
                
                if is_anterior:
                    if gyral_id not in results['anterior_gyri']:
                        results['anterior_gyri'].append(gyral_id)
                else:
                    if gyral_id not in results['posterior_gyri']:
                        results['posterior_gyri'].append(gyral_id)
        
        # Save new labels to subject 
        if load_results: 
            if results['anterior_gyri']:
                anterior_vertices = []
                for gyral_id in results['anterior_gyri']:
                    anterior_vertices.extend(results['gyral_clusters'][gyral_id]['vertices'])
                anterior_vertices = np.unique(anterior_vertices)
                anterior_ras = self._subject.surface_RAS[anterior_vertices]
                
                self._subject.load_label(
                    f"{label_name}_anterior_gyri", 
                    label_idxs=anterior_vertices, 
                    label_RAS=anterior_ras
                )
                
                print(f"Created anterior gyri label with {len(anterior_vertices)} vertices")
            
            if results['posterior_gyri']:
                posterior_vertices = []
                for gyral_id in results['posterior_gyri']:
                    posterior_vertices.extend(results['gyral_clusters'][gyral_id]['vertices'])
                posterior_vertices = np.unique(posterior_vertices)
                posterior_ras = self._subject.surface_RAS[posterior_vertices]
                
                self._subject.load_label(
                    f"{label_name}_posterior_gyri", 
                    label_idxs=posterior_vertices, 
                    label_RAS=posterior_ras
                )
                
                print(f"Created posterior gyri label with {len(posterior_vertices)} vertices")
        
        return results
    
    def label_centroid(self, label_name: str, load: bool = True, centroid_face: bool = False, 
                       custom_vertexes: Optional[np.ndarray] = None) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Compute the centroid of a label. Computes weighted average of vertices in the label 
        and finds the closest surface vertex to the centroid.

        Parameters:
            label_name (str): Name of the label.
            load (bool): If True, load the centroid as a new label.
            centroid_face (bool): If True, return the centroid faces associated with the centroid.
            custom_vertexes (np.ndarray): Custom vertex indices to compute centroid from.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], np.ndarray]: 
                Either the centroid vertex and RAS coordinates, or just the RAS coordinates.
        """
        # Identify vertices
        if custom_vertexes is not None:
            if custom_vertexes.ndim == 2:  # If these are RAS coordinates
                # Find closest surface vertex for each RAS coordinate
                nearest_vertices = []
                for ras in custom_vertexes:
                    nearest_vertex = surface_utils.find_closest_vertex(ras, self._subject.surface_RAS)[0]
                    nearest_vertices.append(nearest_vertex)
                label_vertices = np.unique(nearest_vertices)
            else:  # If these are vertex indices
                label_vertices = custom_vertexes
                
            label_faces = surface_utils.get_faces_from_vertices(self._subject.faces, label_vertices)
        else:
            if label_name not in self._subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            label_vertices = self._subject.labels[label_name].vertex_indexes
            label_faces = surface_utils.get_faces_from_vertices(self._subject.faces, label_vertices)

        # Calculate centroid as weighted average of vertices
        centroid_ras = surface_utils.calculate_geometric_centroid(self._subject.surface_RAS, label_faces)
        
        # Find closest surface vertex to the centroid
        centroid_surface_vertex = surface_utils.find_closest_vertex(centroid_ras, self._subject.surface_RAS)[0]
        centroid_surface_ras = self._subject.surface_RAS[centroid_surface_vertex]

        if centroid_face:
            # Get faces that contain the centroid vertex
            centroid_faces = np.where(np.any(self._subject.faces == centroid_surface_vertex, axis=1))[0]
            centroid_faces_vertices = np.unique(self._subject.faces[centroid_faces])
            centroid_faces_ras = self._subject.surface_RAS[centroid_faces_vertices]
            
            if load:
                self._subject.load_label(
                    f'{label_name}_centroid', 
                    label_idxs=centroid_faces_vertices, 
                    label_RAS=centroid_faces_ras
                )
            
            return centroid_faces_vertices, centroid_faces_ras
        else:
            # Just return the centroid vertex
            centroid_surface_vertex = np.array([centroid_surface_vertex])
            
            if load:
                self._subject.load_label(
                    f'{label_name}_centroid', 
                    label_idxs=centroid_surface_vertex, 
                    label_RAS=np.array([centroid_surface_ras])
                )
            
            return centroid_surface_ras
    
    def get_deepest_sulci(self, percentage: float = 10, label_name: Optional[str] = None, 
                         return_mask: bool = False, load_label: bool = False, 
                         result_label_name: Optional[str] = None) -> np.ndarray:
        """
        Returns indices or a mask of vertices corresponding to the specified percentage 
        of deepest sulci (highest sulc values), either within a specific label or across the entire brain.
        
        Parameters:
        -----------
        percentage : float
            Percentage of deepest sulci to include (default: 10)
        label_name : str or None
            If provided, looks for deepest sulci within this label; if None, looks across the whole brain
        return_mask : bool
            If True, returns a boolean mask; if False, returns indices (default: False)
        load_label : bool
            If True, creates a label for the deepest sulci (default: False)
        result_label_name : str
            Name for the created label if load_label is True
            
        Returns:
        --------
        numpy.ndarray
            Either boolean mask where True values represent vertices in the deepest sulci,
            or array of vertex indices of the deepest sulci
        """
        # Get all sulcal depth values
        sulc = self.sulc_vals
        
        if label_name is not None:
            # Check if the label exists
            if label_name not in self._subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            # Get vertex indices from the label
            label_indices = self._subject.labels[label_name].vertex_indexes
            
            # Extract sulc values only for these vertices
            label_sulc = sulc[label_indices]
            
            # Calculate the threshold value for the top percentage of deepest sulci within this label
            threshold = np.percentile(label_sulc, 100 - percentage)
            
            # Create a mask for vertices in this label that meet the threshold
            label_mask = label_sulc >= threshold
            
            # Map back to global vertex indices
            deepest_indices = label_indices[label_mask]
            
            # Create a global mask (all False initially)
            global_mask = np.zeros(len(sulc), dtype=bool)
            global_mask[deepest_indices] = True
            
            mask = global_mask
        else:
            # Working with the entire brain
            # Calculate the threshold value for the top percentage of deepest sulci
            threshold = np.percentile(sulc, 100 - percentage)
            
            # Create the mask
            mask = sulc >= threshold
            
            deepest_indices = np.where(mask)[0]
        
        # Optionally create a label
        if load_label:
            if result_label_name is None:
                if label_name is not None:
                    result_label_name = f'{label_name}_deepest_{percentage}'
                else:
                    result_label_name = f'deepest_sulci_{percentage}'
            
            # Get RAS coordinates for the deepest sulci
            deepest_ras = self._subject.surface_RAS[deepest_indices]
            
            # Create a label with the deepest sulci
            self._subject.load_label(
                label_name=result_label_name, 
                label_idxs=deepest_indices, 
                label_RAS=deepest_ras, 
                label_stat=sulc[deepest_indices]
            )
        
        if return_mask:
            return mask
        else:
            return deepest_indices
        
    def threshold_label(self, label_name: str, threshold_type: str = 'absolute', 
                   threshold_direction: str = '>=', threshold_value: float = 0, 
                   threshold_measure: str = 'sulc', load_label: bool = False, 
                   new_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Threshold a label or the entire cortex based on a statistical value.

        Parameters:
            label_name (str): Name of the label, or 'cortex' to use the entire brain.
            threshold_type (str): Type of threshold - 'absolute' or 'percentile'.
            threshold_threshold_direction (str): Direction of thresholding - '>', '>=', '<', or '<='.
            threshold_value (float): Value to threshold at (absolute value or percentile).
            threshold_measure (str): Measure to threshold on - 'sulc', 'thickness', 'curv', or 'label_stat'.
            load_label (bool): If True, load the thresholded result as a new label.
            new_name (str): Name for the thresholded label if load_label is True.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Vertex indexes, RAS coordinates, and 
            measurement values of the thresholded result.
        """
        # Determine the set of vertices to consider (specific label or whole cortex)
        if label_name == 'cortex':
            # Use all vertices in the brain
            all_vertices = self._subject.vertex_indexes
            all_ras = self._subject.surface_RAS
            
            # Determine the measurement values to use for thresholding
            if threshold_measure == 'sulc':
                measure_values = self.sulc_vals
            elif threshold_measure == 'thickness':
                measure_values = self.thickness
            elif threshold_measure == 'curv':
                measure_values = self.curv
            else:
                raise ValueError(f"For 'cortex', threshold_measure must be 'sulc', 'thickness', or 'curv'")
            
            # Create a dictionary mapping vertex indices to their measure values
            vertex_to_measure = {vertex: measure_values[vertex] for vertex in all_vertices}
            
        else:
            # Check if the label exists
            if label_name not in self._subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            # Get the label
            label = self._subject.labels[label_name]
            all_vertices = label.vertex_indexes
            all_ras = label.ras_coords
            
            # Determine the measurement values to use for thresholding
            if threshold_measure == 'label_stat':
                if label.label_stat is None:
                    raise ValueError(f"Label '{label_name}' does not have statistical values for thresholding")
                
                # For label_stat, we need to keep track of indices in the label's array
                measure_values = label.label_stat
                vertex_to_measure = {i: value for i, value in enumerate(measure_values)}
                
            elif threshold_measure == 'sulc':
                measure_values = np.array([self.sulc_vals[vertex] for vertex in all_vertices])
                vertex_to_measure = {i: value for i, value in enumerate(measure_values)}
                
            elif threshold_measure == 'thickness':
                measure_values = np.array([self.thickness[vertex] for vertex in all_vertices])
                vertex_to_measure = {i: value for i, value in enumerate(measure_values)}
                
            elif threshold_measure == 'curv':
                measure_values = np.array([self.curv[vertex] for vertex in all_vertices])
                vertex_to_measure = {i: value for i, value in enumerate(measure_values)}
                
            else:
                raise ValueError(f"threshold_measure must be 'sulc', 'thickness', 'curv', or 'label_stat'")
        
        # Calculate the threshold value if using percentile
        if threshold_type == 'percentile':
            if threshold_value < 0 or threshold_value > 100:
                raise ValueError("Percentile threshold must be between 0 and 100")
            
            threshold = np.percentile(list(vertex_to_measure.values()), threshold_value)
        elif threshold_type == 'absolute':
            threshold = threshold_value
        else:
            raise ValueError("threshold_type must be 'percentile' or 'absolute'")
        
        # Apply the thresholding operation based on the threshold_direction
        if threshold_direction == '>':
            mask = np.array([vertex_to_measure[i] > threshold for i in range(len(all_vertices))])
        elif threshold_direction == '>=':
            mask = np.array([vertex_to_measure[i] >= threshold for i in range(len(all_vertices))])
        elif threshold_direction == '<':
            mask = np.array([vertex_to_measure[i] < threshold for i in range(len(all_vertices))])
        elif threshold_direction == '<=':
            mask = np.array([vertex_to_measure[i] <= threshold for i in range(len(all_vertices))])
        else:
            raise ValueError("threshold_direction must be '>', '>=', '<', or '<='")
        
        # Get the thresholded vertices, RAS coordinates, and measure values
        thresholded_indices = np.where(mask)[0]
        
        if label_name == 'cortex':
            thresholded_vertices = all_vertices[thresholded_indices]
            thresholded_ras = all_ras[thresholded_vertices]
            thresholded_values = np.array([measure_values[vertex] for vertex in thresholded_vertices])
        else:
            thresholded_vertices = all_vertices[thresholded_indices]
            thresholded_ras = all_ras[thresholded_indices]
            thresholded_values = measure_values[thresholded_indices]
        
        # Optionally create a new label
        if load_label:
            if new_name is None:
                threshold_direction_str = threshold_direction.replace('>', 'gt').replace('<', 'lt').replace('=', 'eq')
                type_str = 'pct' if threshold_type == 'percentile' else 'abs'
                new_name = f"{label_name}_{threshold_measure}_{threshold_direction_str}_{threshold_value}_{type_str}"
            
            self._subject.load_label(
                label_name=new_name, 
                label_idxs=thresholded_vertices, 
                label_RAS=thresholded_ras, 
                label_stat=thresholded_values
            )
        
        return thresholded_vertices, thresholded_ras, thresholded_values