from functools import cached_property
from typing import List, Tuple, Union, Callable
import numpy as np
import nibabel as nb
from pathlib import Path
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.spatial.distance import cdist

import trimesh as tm

from scalpel.classes.label import Label
from scalpel.utilities import freesurfer_utils as fsu
from scalpel.utilities import surface_utils
from scalpel.utilities.plotting import initialize_scene, plot, plot_label, remove_label, show_scene

class ScalpelSubject(object):
    def __init__(self, subject_id, hemi, subjects_dir, surface_type="inflated"):
        self._subject_id = subject_id
        self._hemi = hemi
        self._surface_type = surface_type
        self._labels = defaultdict(list)
        self._subject_fs_path = Path(f'{subjects_dir}/{subject_id}/')
        self._subjects_dir = subjects_dir
        self._scene = None
        self._mesh = {}
    
        surface = nb.freesurfer.read_geometry(f'{subjects_dir}/{subject_id}/surf/{hemi}.{surface_type}')
        self._curv = nb.freesurfer.read_morph_data(f'{subjects_dir}/{self._subject_id}/surf/{self._hemi}.curv')
        self._ras_coords, self._faces = surface[0], surface[1]
        self._gyrus = fsu.get_gyrus(np.unique(self._faces), self._ras_coords, self._curv)
        self._sulcus = fsu.get_sulcus(np.unique(self._faces), self._ras_coords, self._curv)

    ############################
    # Properties
    ############################    
    @property
    def subject_id(self):
        ## Subject ID
        return self._subject_id

    @property
    def hemi(self):
        ## Hemisphere
        return self._hemi

    @property
    def subjects_dir(self):
        ## Subjects Directory
        return self._subjects_dir

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

    @cached_property
    def pial_v(self):
        ## Pial surface vertices | required to get sulcal depth
        pial_path = f'{self.subject_fs_path}/surf/{self.hemi}.pial'
        pial_verts, _ = nb.freesurfer.read_geometry(pial_path)
        return pial_verts
    
    @cached_property
    def gyrif_v(self):
        ## Gyrus-inflated surface vertices (pial-outer-smoothed)
        gyrif_path = f'{self.subject_fs_path}/surf/{self.hemi}.pial-outer-smoothed'
        gyrif_verts, _ = nb.freesurfer.read_geometry(gyrif_path)
        return gyrif_verts
    
    @cached_property
    def sulc_vals(self):
        ## Returns vertex-wise sulc values
        sulc_path = f'{self.subject_fs_path}/surf/{self.hemi}.sulc'
        sulc_vals= nb.freesurfer.read_morph_data(sulc_path)
        return sulc_vals

    @property
    def subject_fs_path(self):
        ## Path to subject's freesurfer directory
        return self._subject_fs_path

    @cached_property
    def mesh(self):
        gyrus_gray = [250, 250, 250]
        sulcus_gray = [130, 130, 130]
        if self.surface_type == 'inflated':
            print('Initial plot builds cortical mesh (~1 minute)')
            gyrus_mesh = surface_utils.make_mesh(self._ras_coords, self._faces, self._gyrus[0], face_colors=gyrus_gray)
            sulcus_mesh = surface_utils.make_mesh(self._ras_coords, self._faces, self._sulcus[0], face_colors=sulcus_gray, include_all = True)
            self._mesh['gyrus'] = gyrus_mesh
            self._mesh['sulcus'] = sulcus_mesh
        else:
            self._mesh['cortex'] = surface_utils.make_mesh(self._ras_coords, self._faces, self.vertex_indexes, face_colors=gyrus_gray, include_all=True)
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

    def load_label(self, label_name, label_idxs=None, label_RAS=None, label_stat=None, custom_label_path=None):
        """
        Load a label into the subject class. Either loads from file or from input parameters.

        Parameters:
        - label_name (str): Name of the label.
        - label_idxs (np.array, optional): Vertex indices of the label. Defaults to None.
        - label_RAS (np.array, optional): RAS coordinates of the label. Defaults to None.
        - label_stat (np.array, optional): Statistical values of the label. Defaults to None.
        - custom_label_path (str, optional): Path to a custom label file. Defaults to None.

        Returns:
        - None 
        """
        if label_idxs is None or label_RAS is None:
            if custom_label_path is None:
                self._labels[label_name] = Label(label_name, self.hemi, subject_id=self.subject_id, subjects_dir = self._subjects_dir, custom_label_path = f'{self.subject_fs_path}/label/{self.hemi}.{label_name}.label')
    
            else:
                if isinstance(custom_label_path, str):
                    custom_label_path = Path(custom_label_path)
                self._labels[label_name] = Label(label_name, subject_id=self.subject_id, subjects_dir = self._subjects_dir,  hemi = self.hemi, custom_label_path = custom_label_path)
        else:
            self._labels[label_name] = Label(label_name, subject_id = self.subject_id, subjects_dir = self._subjects_dir,  hemi = self.hemi, vertex_indexes = label_idxs, ras_coords = label_RAS, stat = label_stat)
    
    def remove_label(self, label_name):
        """
        Remove a label from the subject class.

        Parameters:
        - label_name (str): Name of the label.

        Returns:
        - None
        """
        self._labels.pop(label_name)

    def write_label(self, label_name: str, save_label_name: str = None, custom_label_dir: str = None, overwrite = False):
        """
        Write a label to a file.

        Parameters:
        - label_name (str): Name of the label.
        - save_label_name (str): Name of the label to save.
        - custom_label_path (str): Path to save the label.

        Returns:
        - None

        """


        if custom_label_dir is not None:
            label_dir_path = Path(custom_label_dir)
        else:
            label_dir_path = self.subject_fs_path / "label"
        
        if save_label_name is None:
            save_label_name = f"{self.hemi}.{label_name}.label"

        self._labels[label_name].write_label(label_name = save_label_name, label_dir_path = label_dir_path, overwrite = overwrite)

    ############################
    # Visualization Methods
    ############################

    def plot(self, view='lateral', labels: List[str] = None):
        if self._mesh == {}:
            self.mesh
        if self._scene is None:
            self._scene = initialize_scene(self._mesh, view, self._hemi, self._surface_type)
        return plot(scene = self._scene, hemi = self.hemi, view = view, labels = labels)

    def plot_label(self, label_name: str, view='lateral', label_ind=None, face_colors=None):
        assert label_name in self.labels, f"Label {label_name} not found in subject {self.subject_id}"
        if self._scene is None:
            if self._mesh == {}:
                self.mesh
            self._scene = initialize_scene(self._mesh, view, self._hemi, self._surface_type)
        return plot_label(self._scene, self._ras_coords, self._faces, self._labels, label_name, view, self._hemi, face_colors, label_ind)

    def remove_label(self, label_name: str):
        return remove_label(self._scene, label_name)

    def show(self):
        return show_scene(self._scene)

    def create_subject(name, hemi="lh", surface_type="inflated"):
        subject = ScalpelSubject(name, hemi, surface_type)
        return subject
    
    def save_plot(self, filename: str, save_dir = None, distance = 500, resolution = 'low'):
        """
        Save the trimesh Scene to file
        """
        if self._scene is None:
            raise ValueError("Scene not initialized. Please call plot() first.")
        import io
        from PIL import Image

        if resolution == 'low':
            resolution = (512, 512)
        elif resolution == 'medium':
            resolution = (720, 720)
        elif resolution == 'high':
            resolution = (1080, 1080)
        
        # Set a proper camera distance to capture the entire mesh
        self._scene.set_camera(distance=distance)  
        
        if save_dir is not None:
            filename = Path(save_dir) / filename
        
        data = self._scene.save_image(resolution=resolution)
        image = Image.open(io.BytesIO(data))
        image.save(filename)
        #
          
    
    ############################
    # Gyral Analysis Methods
    ############################

    def combine_labels(self, label_names: List[str], new_label_name: str) -> None:
        """
        Combine multiple labels into a single new label.

        Parameters:
            label_names (List[str]): List of label names to combine.
            new_label_name (str): Name for the new combined label.

        Raises:
            ValueError: If any of the input labels don't exist.
        """
        if not all(name in self.labels for name in label_names):
            raise ValueError("All input labels must exist in the subject.")

        combined_ind = np.unique(np.concatenate([self.labels[name].vertex_indexes for name in label_names])).astype(int)
        combined_ras = self.ras_coords[combined_ind]
        self.load_label(new_label_name, combined_ind, combined_ras)

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
        if label_name not in self.labels:
            raise ValueError(f"Label '{label_name}' does not exist.")

        label_faces = surface_utils.get_faces_from_vertices(self.faces, self.labels[label_name].vertex_indexes)
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

        Parameters:
            label1 (str): Name of the first label.
            label2 (str): Name of the second label.

        Returns:
            np.ndarray: Indices of shared gyral clusters.

        Raises:
            ValueError: If either label doesn't exist.
        """
        if label1 not in self.labels or label2 not in self.labels:
            raise ValueError("Both labels must exist in the subject.")

        label1_faces = surface_utils.get_faces_from_vertices(self.faces, self.labels[label1].vertex_indexes, include_all = False)
        label2_faces = surface_utils.get_faces_from_vertices(self.faces, self.labels[label2].vertex_indexes, include_all = False)

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

        Parameters:
            shared_gyral_clusters (np.ndarray): Array of shared gyral cluster indices.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices and RAS coordinates of the shared gyral region.
        """
        shared_gyral_mask = np.isin(self.gyral_clusters, shared_gyral_clusters)
        shared_gyral_index = self.gyrus[0][shared_gyral_mask]
        shared_gyral_ras = self.ras_coords[shared_gyral_index]
        return shared_gyral_index, shared_gyral_ras

    def find_gyral_gap(self, label1: str, label2: str, method: str = 'pca', n_components: int = 2, 
                       n_clusters: Union[int, List[int]] = [2, 3], clustering_algorithm: str = 'agglomerative',
                       disjoints = True) -> dict:
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

        if label1 not in self.labels or label2 not in self.labels:
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
        if disjoints:
            shared_gyral_faces = surface_utils.get_faces_from_vertices(self.faces, shared_index)
            disjoints = surface_utils.get_label_subsets(shared_gyral_faces, self.faces)

            disjoints.sort(key=lambda x: len(x), reverse=True)
            shared_index = np.unique(disjoints[0])
            shared_ras = self.ras_coords[shared_index]

        return {
            'label1_analysis': analysis1,
            'label2_analysis': analysis2,
            'closest_clusters': (closest1, closest2),
            'min_cluster_distance': min_distance,
            'shared_gyral_clusters': shared_clusters,
            'shared_gyral_index': shared_index,
            'shared_gyral_ras': shared_ras
        }
    def analyze_sulcal_gyral_relationships(self, label_name, gyral_clusters=300, sulcal_clusters=None, 
                                      algorithm='kmeans', load_results=True):
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
        if sulcal_clusters is None:
            sulcal_clusters = self.subject.labels[label_name].vertex_indexes.shape[0] // 400

       
        if label_name not in self.labels:
            raise ValueError(f"Sulcal label '{label_name}' not found in self")
        
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
            cluster_ras = self.ras_coords[cluster_vertices]
            centroid = np.mean(cluster_ras, axis=0)
            
            results['gyral_clusters'][cluster_id] = {
                'vertices': cluster_vertices,
                'ras_coords': cluster_ras,
            }
        
        # Cluster the sulcus
        sulcus_vertices = self.labels[label_name].vertex_indexes
        sulcus_ras = self.ras_coords[sulcus_vertices]
        
        
        if algorithm == 'kmeans':
            from sklearn.cluster import KMeans
            sulcal_clustering = KMeans(n_clusters=sulcal_clusters, random_state=42, n_init="auto")
        elif algorithm == 'agglomerative':
            from sklearn.cluster import AgglomerativeClustering
            sulcal_clustering = AgglomerativeClustering(n_clusters=sulcal_clusters)
        elif algorithm == 'dbscan':
            from sklearn.cluster import DBSCAN
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
                self.load_label(cluster_label_name, 
                                label_idxs=cluster_vertices, 
                                label_RAS=cluster_ras)
            
            results['sulcal_clusters'][cluster_id] = {
                'vertices': cluster_vertices,
                'ras_coords': cluster_ras,
                'centroid': centroid
            }
        
        # For each sulcal cluster, find adjacent gyral clusters
        for sulcal_id in results['sulcal_clusters']:
           
            sulcal_cluster_vertices = results['sulcal_clusters'][sulcal_id]['vertices']
            
            
            sulcal_faces = surface_utils.get_faces_from_vertices(self.faces, sulcal_cluster_vertices)
            sulcal_boundary = surface_utils.find_label_boundary(sulcal_faces)
            
            
            boundary_neighbors = set()
            for face in self.faces:
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
            
            #  Compare centroid RAS to determine anterior/posterior relationship
            sulcal_centroid = self.label_centroid(label_name, load = False, custom_vertexes = results['sulcal_clusters'][sulcal_id]['ras_coords'])
            
            for gyral_id in adjacent_gyral_clusters:
                gyral_centroid = self.label_centroid(label_name, load = False, custom_vertexes = results['gyral_clusters'][gyral_id]['ras_coords'])
                
                
                is_anterior = gyral_centroid[1] > sulcal_centroid[1]
                
                if is_anterior:
                    if gyral_id not in results['anterior_gyri']:
                        results['anterior_gyri'].append(gyral_id)
                else:
                    if gyral_id not in results['posterior_gyri']:
                        results['posterior_gyri'].append(gyral_id)
        
        # save new labels to subject 
        if load_results: 
            if results['anterior_gyri']:
                anterior_vertices = []
                for gyral_id in results['anterior_gyri']:
                    anterior_vertices.extend(results['gyral_clusters'][gyral_id]['vertices'])
                anterior_vertices = np.unique(anterior_vertices)
                anterior_ras = self.ras_coords[anterior_vertices]
                
                self.load_label(f"{label_name}_anterior_gyri", 
                                label_idxs=anterior_vertices, 
                                label_RAS=anterior_ras)
                
                print(f"Created anterior gyri label with {len(anterior_vertices)} vertices")
            
            
            if results['posterior_gyri']:
                posterior_vertices = []
                for gyral_id in results['posterior_gyri']:
                    posterior_vertices.extend(results['gyral_clusters'][gyral_id]['vertices'])
                posterior_vertices = np.unique(posterior_vertices)
                posterior_ras = self.ras_coords[posterior_vertices]
                
                self.load_label(f"{label_name}_posterior_gyri", 
                                label_idxs=posterior_vertices, 
                                label_RAS=posterior_ras)
                
                print(f"Created posterior gyri label with {len(posterior_vertices)} vertices")
        
        return results
    
    def label_centroid(self, label_name: str, load = True, centroid_face = False, custom_vertexes = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the centroid of a label. Computes weighted average of vertices in the label and finds the closest surface vertex to the centroid.

        Parameters:
            label_name (str): Name of the label.
            centroid_face (bool): If True, return the centroid faces assopciated with the centroid. Defaults to False.

        Returns:
            np.ndarray: Centroid coordinates.
        """
        # Identify vertices
        if custom_vertexes is not None:
            label_faces = surface_utils.get_faces_from_vertices(self.faces, custom_vertexes)
            label_faces_ind = np.where(np.isin(self.faces, custom_vertexes))[0] 
            label_faces = self.faces[label_faces_ind]
        else: 

            label_faces_ind = np.where(np.isin(self.faces, self.labels[label_name].vertex_indexes))[0]
            label_faces = self.faces[label_faces_ind]

        # Calculate centroid as weighted average of vertices
        centroid_ras = surface_utils.calculate_geometric_centroid(self.ras_coords, label_faces)
        # Find closest surface vertex to the centroid
        centroid_surface_vertex = surface_utils.find_closest_vertex(centroid_ras, self.ras_coords)[0]
        centroid_surface_ras = self.ras_coords[centroid_surface_vertex]

        if centroid_face:
            centroid_faces = self.faces[np.where(np.isin(self.faces, centroid_surface_vertex))[0]]
            centroid_faces_vertices = np.array([np.unique(self.faces[centroid_faces])])
            centroid_ras = np.array([self.ras_coords[centroid_faces]])
            if load:
                self.load_label(f'{label_name}_centroid', label_idxs=centroid_faces_vertices, label_RAS=centroid_ras)
            return centroid_faces_vertices, centroid_ras
        else:
            centroid_surface_vertex = np.array(centroid_surface_vertex)
            centroid_surface_ras = np.array(centroid_surface_ras)
            if load:
                self.load_label(f'{label_name}_centroid', label_idxs=centroid_surface_vertex, label_RAS=centroid_surface_ras)
            return centroid_surface_vertex, centroid_surface_ras
        
        
    def threshold_label(self, label_name, threshold, load_label = False, new_name = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Threshold a label based on a statistical value.

        Parameters:
            label_name (str): Name of the label.
            threshold (float): Threshold value.
            load_label (bool): If True, load the thresholded label. Defaults to False.
            new_name (str): Name for the thresholded label. Defaults to None.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Vertex indexes, RAS coordinates, and statistical values of the thresholded label.
        
        """


        try:
            thresh_idx = self.labels[label_name].label_stat[self.labels[label_name].label_stat > threshold].index.to_numpy()
        except ValueError:
            print(f"Label {label_name} not found")
            
        
        if load_label:
            if new_name is None:
                new_name = f'{label_name}_{threshold}'
            self.load_label(label_name = new_name, label_idxs=self.labels[label_name].vertex_indexes[thresh_idx], 
                            label_RAS=self.labels[label_name].ras_coords[thresh_idx], label_stat=self.labels[label_name].label_stat[thresh_idx])
        
        return self.labels[label_name].vertex_indexes[thresh_idx], self.labels[label_name].ras_coords[thresh_idx], self.labels[label_name].label_stat[thresh_idx]
    
    def get_deepest_sulci(self, percentage=10, label_name=None, return_mask=False, load_label=False, result_label_name=None):
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
            Name for the created label if load_label is True (default: 'deepest_sulci_{percentage}' or '{label_name}_deepest_{percentage}')
            
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
            if label_name not in self.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            # Get vertex indices from the label
            label_indices = self.labels[label_name].vertex_indexes
            
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
            deepest_ras = self.ras_coords[deepest_indices]
            
            # Create a label with the deepest sulci
            self.load_label(
                label_name=result_label_name, 
                label_idxs=deepest_indices, 
                label_RAS=deepest_ras, 
                label_stat=sulc[deepest_indices]
            )
        
        if return_mask:
            return mask
        else:
            # Return the indices of vertices that meet the threshold
            return deepest_indices
    ######
    # Sulcal Measurements
    ######
    
    def calculate_sulcal_depth(self, label_name, depth_pct=8, n_deepest=100, use_n_deepest=True):
        """
        Calculate the depth of a sulcus matching the MATLAB calcSulc_depth function.
        
        Parameters:
        -----------
        label_name: str
            Name of the label corresponding to the sulcus
        depth_pct: float
            Percentage of deepest vertices to use (default: 8, matching MATLAB default)
        n_deepest: int
            Number of deepest vertices to use (default: 100)
        use_n_deepest: bool
            If True, use n_deepest vertices; if False, use depth_pct percentage (default: True)
                
        Returns:
        --------
        float: The median depth of the sulcus in mm

        NOTE: Rquires the pial and gyral-inflated surfaces to be generated with recon-all -all
        """
        try:
            if label_name not in self._labels:
                raise ValueError(f"Label '{label_name}' not found")
                    
            label_vertices = self._labels[label_name].vertex_indexes
                
            if not isinstance(label_vertices, np.ndarray):
                label_vertices = np.array(label_vertices, dtype=int)
                
            sulc_map = self.sulc_vals
            
            label_sulc_values = sulc_map[label_vertices]
                
            sorted_indices = np.argsort(label_sulc_values)
            sorted_sulc = np.sort(label_sulc_values)
            
            
            num_vertices = len(sorted_indices)
            
            if use_n_deepest:
                num_fundus = min(n_deepest, num_vertices) 
            else:
                num_fundus = int(np.ceil(num_vertices * depth_pct / 100))
            
            
            fundus_indices = sorted_indices[-num_fundus:]
            fundus_vertices = label_vertices[fundus_indices]
                
            # Calculate distances from pial to gyral-inflated surface
            depths = []
            for vertex_idx in fundus_vertices:
                # Get coordinates of the vertex on the pial surface
                v_xyz = self.pial_v[vertex_idx]
                    
                # Calculate distances to all gyral-inflated vertices
                # NOTE: The gyral-inflated surface is generated with recon-all flag -all
                distances = np.sqrt(np.sum((self.gyrif_v - v_xyz)**2, axis=1))
                    
                # Find minimum distance
                min_distance = np.min(distances)
                depths.append(min_distance)
                
            # Return median depth
            if len(depths) > 0:
                return np.median(depths)
            else:
                return np.nan
                
        except Exception as e:
            print(f"Error calculating sulcal depth for {label_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.nan
