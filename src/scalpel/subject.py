from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
from functools import cached_property
import nibabel as nib
import numpy as np
from collections import defaultdict

# Import the modules we need
from scalpel.analysis.analyzer import ScalpelAnalyzer
from scalpel.visualization.visualizer import ScalpelVisualizer
from scalpel.measurement.measurer import ScalpelMeasurer
from scalpel.classes.label import Label

class ScalpelSubject:
    """
    Core class for CNL_scalpel
    
    This class represents a FreeSurfer subject and provides access to
    surface data, labels, and other relevant information. It delegates
    visualization tasks to a ScalpelVisualizer instance, analysis tasks
    to a ScalpelAnalyzer instance, and measurement tasks to a ScalpelMeasurer
    instance.
    """
    
    def __init__(self, subject_id: str, hemi: str, subjects_dir: Union[str, Path], surface_type: str = 'inflated'):
        """
        Initialize a ScalpelSubject.
        
        Parameters:
        -----------
        subject_id : str
            FreeSurfer subject ID
        hemi : str
            Hemisphere ('lh' or 'rh')
        subjects_dir : Union[str, Path]
            Path to FreeSurfer subjects directory
        surface_type : str, default='inflated'
            Surface type to use ('inflated', 'pial', 'white', etc.)
        """
        self._subject_id = subject_id
        self._hemi = hemi
        self._surface_type = surface_type
        self._subjects_dir = subjects_dir if isinstance(subjects_dir, Path) else Path(subjects_dir)
        self._subject_fs_path = self._subjects_dir / subject_id
        self._labels = defaultdict(list)
        
        # Validate that the subject directory exists
        assert self.subject_fs_path.exists(), f"Subject path does not exist at {self.subject_fs_path}"
        
        # Load the surface
        try:
            surface_path = self._subject_fs_path / 'surf' / f'{hemi}.{surface_type}'
            self._surface = nib.freesurfer.read_geometry(str(surface_path))
        except ValueError:
            print(f"Surface not found at {self._subject_fs_path}/surf/{hemi}.{surface_type}")
            raise

    ############################
    # Properties
    ############################
    
    @property
    def subject_id(self):
        """Subject ID"""
        return self._subject_id

    @property
    def hemi(self):
        """Hemisphere"""
        return self._hemi

    @property
    def subjects_dir(self):
        """Subjects Directory"""
        return self._subjects_dir

    @property
    def surface_type(self):
        """Surface Type - inflated, pial, white"""
        return self._surface_type
    
    @property
    def subject_fs_path(self):
        """Path to subject's freesurfer directory"""
        return self._subject_fs_path
    
    @cached_property
    def surface(self):
        """Surface - the raw geometry data"""
        return self._surface
    
    @cached_property
    def surface_RAS(self):
        """Right, Anterior, Superior Coordinates of all vertices"""
        return self._surface[0]
    
    @cached_property
    def faces(self):
        """Faces of the mesh"""
        return self._surface[1]
    
    @cached_property
    def vertex_indexes(self):
        """Unique vertex indexes"""
        return np.unique(self.faces)
    
    @property
    def labels(self):
        """
        Access loaded labels through dictionary
    
        sub.labels['label_name'].vertex_indexes  # Vertex indexes
        sub.labels['label_name'].ras_coords      # RAS coordinates
        """
        return self._labels
    
    # Surface loading methods
    def _load_surface(self, surface_name: str):
        """Load a specific surface by name"""
        surface_path = self._subject_fs_path / 'surf' / f'{self._hemi}.{surface_name}'
        if not surface_path.exists():
            raise FileNotFoundError(f"Surface not found: {surface_path}")
        return nib.freesurfer.read_geometry(str(surface_path))
    
    @cached_property
    def white_v(self):
        """White matter surface vertices"""
        white_surface = self._load_surface('white')
        return white_surface[0]  # vertices
    
    @cached_property
    def pial_v(self):
        """Pial surface vertices"""
        pial_surface = self._load_surface('pial')
        return pial_surface[0]  # vertices
    
    @cached_property
    def gyrif_v(self):
        """Gyral-inflated surface vertices (requires recon-all -all)"""
        try:
            gyrif_surface = self._load_surface('inflated')  # or 'sphere.reg' depending on your setup
            return gyrif_surface[0]  # vertices
        except FileNotFoundError:
            print(f"Warning: Gyral-inflated surface not found for {self._hemi}. Using inflated surface.")
            return self.surface_RAS
    
    # Curvature and morphometric data loading
    def _load_curv_file(self, curv_name: str):
        """Load a curvature file"""
        curv_path = self._subject_fs_path / 'surf' / f'{self._hemi}.{curv_name}'
        if not curv_path.exists():
            raise FileNotFoundError(f"Curvature file not found: {curv_path}")
        return nib.freesurfer.read_morph_data(str(curv_path))
    
    @cached_property
    def mean_curvature(self):
        """Mean curvature values from FreeSurfer .curv file"""
        return self._load_curv_file('curv')
    
    @cached_property
    def gaussian_curvature(self):
        """Gaussian curvature values"""
        try:
            return self._load_curv_file('curv.K')
        except FileNotFoundError:
            print(f"Warning: Gaussian curvature file not found for {self._hemi}")
            return np.zeros(len(self.surface_RAS))
    
    @cached_property
    def thickness(self):
        """Cortical thickness values from FreeSurfer .thickness file"""
        return self._load_curv_file('thickness')
    
    @cached_property
    def sulc_vals(self):
        """Sulcal depth values from FreeSurfer .sulc file"""
        return self._load_curv_file('sulc')
    
    @cached_property
    def curv(self):
        """Curvature values from FreeSurfer .curv file (alias for mean_curvature)"""
        return self.mean_curvature
    
    # Derived properties based on curvature
    @cached_property
    def gyrus(self):
        """Get gyral vertices based on curvature (negative curvature)"""
        return np.where(self.mean_curvature < 0)[0]
    
    @cached_property
    def sulcus(self):
        """Get sulcal vertices based on curvature (positive curvature)"""
        return np.where(self.mean_curvature > 0)[0]
    
    @cached_property
    def plotter(self):
        """Get the ScalpelVisualizer instance for this subject"""
        if not hasattr(self, '_plotter'):
            self._plotter = ScalpelVisualizer(self)
        return self._plotter
    
    @cached_property
    def analyzer(self):
        """Get the ScalpelAnalyzer instance for this subject"""
        if not hasattr(self, '_analyzer'):
            self._analyzer = ScalpelAnalyzer(self)
        return self._analyzer
    
    @cached_property
    def measurer(self):
        """Get the ScalpelMeasurer instance for this subject"""
        if not hasattr(self, '_measurer'):
            self._measurer = ScalpelMeasurer(self)
        return self._measurer
    
    @property
    def gyral_clusters(self):
        """Perform K-means clustering on gyral regions."""
        return self.analyzer.gyral_clusters
    
    @cached_property
    def adjacency_matrix(self):
        return self.analyzer.adjacency
    
    ############################
    # Label Management
    ############################
    
    def load_label(self, label_name: str, label_idxs=None, label_RAS=None, label_stat=None, custom_label_path=None):
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
                self._labels[label_name] = Label(
                    label_name, 
                    self.hemi, 
                    subject_id=self.subject_id, 
                    subjects_dir=self._subjects_dir, 
                    custom_label_path=f'{self.subject_fs_path}/label/{self.hemi}.{label_name}.label'
                )
            else:
                if isinstance(custom_label_path, str):
                    custom_label_path = Path(custom_label_path)
                self._labels[label_name] = Label(
                    label_name, 
                    subject_id=self.subject_id, 
                    subjects_dir=self._subjects_dir, 
                    hemi=self.hemi, 
                    custom_label_path=custom_label_path
                )
        else:
            self._labels[label_name] = Label(
                label_name, 
                subject_id=self.subject_id, 
                subjects_dir=self._subjects_dir, 
                hemi=self.hemi, 
                vertex_indexes=label_idxs, 
                ras_coords=label_RAS, 
                stat=label_stat
            )
    
    def remove_label(self, label_name: str):
        """
        Remove a label from the subject class.

        Parameters:
        - label_name (str): Name of the label.

        Returns:
        - None
        """
        if label_name in self._labels:
            self._labels.pop(label_name)
    
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
        combined_ras = self.surface_RAS[combined_ind]
        self.load_label(new_label_name, combined_ind, combined_ras)
    
    def write_label(self, label_name: str, save_label_name: str = None, custom_label_dir: str = None, overwrite: bool = False):
        """
        Write a label to a file.

        Parameters:
        - label_name (str): Name of the label.
        - save_label_name (str): Name of the label to save.
        - custom_label_dir (str): Path to save the label.
        - overwrite (bool): Whether to overwrite existing file.

        Returns:
        - None
        """
        if custom_label_dir is not None:
            label_dir_path = Path(custom_label_dir)
        else:
            label_dir_path = self.subject_fs_path / "label"
        
        if save_label_name is None:
            save_label_name = label_name

        self._labels[label_name].write_label(
            label_name=save_label_name, 
            label_dir_path=label_dir_path, 
            overwrite=overwrite
        )
    
    ############################
    # Visualization Methods 
    ############################
    
    def plot(self, view: str = 'lateral', labels: List[str] = None):
        """
        Plot the cortical surface, optionally with labels.
        
        Parameters:
        -----------
        view : str, default='lateral'
            View angle ('lateral', 'medial', 'ventral', 'dorsal')
        labels : List[str], optional
            Names of labels to plot
            
        Returns:
        --------
        trimesh.Scene
            The visualized scene
        """
        return self.plotter.plot(view=view, labels=labels)
    
    def plot_label(self, label_name: str, view: str = 'lateral', label_ind=None, face_colors=None):
        """
        Plot a label on the cortical surface.
        
        Parameters:
        -----------
        label_name : str
            Name of the label to plot
        view : str, default='lateral'
            View angle ('lateral', 'medial', 'ventral', 'dorsal')
        label_ind : np.ndarray, optional
            Vertex indices for the label, if not using a stored label
        face_colors : Union[str, List[int], np.ndarray], optional
            Colors for the label faces
            
        Returns:
        --------
        trimesh.Scene
            The visualized scene
        """
        return self.plotter.plot_label(
            label_name=label_name, 
            view=view, 
            label_ind=label_ind, 
            face_colors=face_colors
        )
    
    def remove_plot_label(self, label_name: str):
        """
        Remove a label from the visualization.
        
        Parameters:
        -----------
        label_name : str
            Name of the label to remove
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        return self.plotter.remove_label(label_name)
    
    def show(self):
        """
        Show the current scene.
        
        Returns:
        --------
        trimesh.Scene
            The visualized scene
        """
        return self.plotter.show()
    
    def save_plot(self, filename: str, save_dir=None, distance: int = 500, resolution: str = 'low'):
        """
        Save the current scene as an image.
        
        Parameters:
        -----------
        filename : str
            Name of the output file
        save_dir : Union[str, Path], optional
            Directory to save the file in
        distance : int, default=500
            Camera distance
        resolution : str, default='low'
            Image resolution ('low', 'medium', 'high')
            
        Returns:
        --------
        None
        """
        return self.plotter.save_plot(
            filename=filename,
            save_dir=save_dir,
            distance=distance,
            resolution=resolution
        )
    
    ############################
    # Analysis Methods 
    ############################
    
    def perform_gyral_clustering(self, n_clusters: int = 300, algorithm: str = 'kmeans'):
        """
        Perform clustering on gyral regions.

        Parameters:
            n_clusters (int): Number of clusters to create.
            algorithm (str): Clustering algorithm to use ('kmeans', 'agglomerative', or 'dbscan').

        Returns:
            np.ndarray: Cluster assignments for gyral vertices.
        """
        return self.analyzer.perform_gyral_clustering(
            n_clusters=n_clusters, 
            algorithm=algorithm
        )
    
    def perform_boundary_analysis(self, label_name: str, method: str = 'pca', 
                                n_components: int = 2, n_clusters: Union[int, List[int]] = [2, 3],
                                clustering_algorithm: str = 'agglomerative'):
        """
        Perform boundary analysis on a label using PCA or direct clustering.

        Parameters:
            label_name (str): Name of the label to analyze.
            method (str): Analysis method ('pca' or 'direct').
            n_components (int): Number of PCA components (ignored if method is 'direct').
            n_clusters (Union[int, List[int]]): Number of clusters for each part of the label.
            clustering_algorithm (str): Clustering algorithm to use.

        Returns:
            dict: Analysis results including cluster assignments.
        """
        return self.analyzer.perform_boundary_analysis(
            label_name=label_name,
            method=method,
            n_components=n_components,
            n_clusters=n_clusters,
            clustering_algorithm=clustering_algorithm
        )
    
    def label_boundary(self, label_name:str, load_label = True):
        """
        Find the boundary vertices and label_RAS for a given label

        Parameters:
            label_name (str): Name of the label to analyze.
            load_label (bool): If True, load the boundary as a new label.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices and RAS coordinates of the label boundary.
        
        """
        boundary = self.analyzer.find_label_boundary(label_name)

        if load_label:
            self.load_label(
                label_name=f'{label_name}_boundary',
                label_idxs=boundary,
                label_RAS=self.surface_RAS[boundary]
            )

        return boundary, self.surface_RAS[boundary]

    def find_closest_clusters(self, analysis_results1: dict, analysis_results2: dict):
        """
        Find the closest clusters between two label analysis results.

        Parameters:
            analysis_results1 (dict): Analysis results for the first label.
            analysis_results2 (dict): Analysis results for the second label.

        Returns:
            Tuple[int, int, float]: Indices of closest clusters and their median distance.
        """
        return self.analyzer.find_closest_clusters(
            analysis_results1=analysis_results1,
            analysis_results2=analysis_results2
        )
    
    def find_shared_gyral_clusters(self, label1: str, label2: str):
        """
        Find shared gyral clusters between two labels.

        Parameters:
            label1 (str): Name of the first label.
            label2 (str): Name of the second label.

        Returns:
            np.ndarray: Indices of shared gyral clusters.
        """
        return self.analyzer.find_shared_gyral_clusters(
            label1=label1,
            label2=label2
        )
    
    def get_shared_gyral_region(self, shared_gyral_clusters: np.ndarray):
        """
        Get the shared gyral region based on shared gyral clusters.

        Parameters:
            shared_gyral_clusters (np.ndarray): Array of shared gyral cluster indices.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices and RAS coordinates of the shared gyral region.
        """
        return self.analyzer.get_shared_gyral_region(
            shared_gyral_clusters=shared_gyral_clusters
        )
    
    def find_gyral_gap(self, label1: str, label2: str, method: str = 'pca', n_components: int = 2, 
                     n_clusters: Union[int, List[int]] = [2, 3], clustering_algorithm: str = 'agglomerative',
                     disjoints: bool = True, load_label = True):
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
        """
        return self.analyzer.find_gyral_gap(
            label1=label1,
            label2=label2,
            method=method,
            n_components=n_components,
            n_clusters=n_clusters,
            clustering_algorithm=clustering_algorithm,
            disjoints=disjoints,
            load_label=load_label
        )
    
    def analyze_sulcal_gyral_relationships(self, label_name: str, gyral_clusters: int = 300, 
                                         sulcal_clusters: Optional[int] = None, 
                                         algorithm: str = 'kmeans', 
                                         load_results: bool = True):
        """
        Comprehensive analysis of sulcal-gyral relationships.

        Parameters:
            label_name (str): Name of the sulcal label
            gyral_clusters (int): Number of clusters for gyral clustering
            sulcal_clusters (int): Number of clusters for sulcal clustering
            algorithm (str): Clustering algorithm ('kmeans', 'agglomerative', or 'dbscan')
            load_results (bool): Whether to load result labels in the subject
            
        Returns:
            dict: Complete analysis results
        """
        return self.analyzer.analyze_sulcal_gyral_relationships(
            label_name=label_name,
            gyral_clusters=gyral_clusters,
            sulcal_clusters=sulcal_clusters,
            algorithm=algorithm,
            load_results=load_results
        )
    
    def label_centroid(self, label_name: str, load: bool = True, centroid_face: bool = False, 
                      custom_vertexes: Optional[np.ndarray] = None):
        """
        Compute the centroid of a label.

        Parameters:
            label_name (str): Name of the label.
            load (bool): If True, load the centroid as a new label.
            centroid_face (bool): If True, return the centroid faces associated with the centroid.
            custom_vertexes (np.ndarray): Custom vertex indices to compute centroid from.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], np.ndarray]: 
                Either the centroid vertex and RAS coordinates, or just the RAS coordinates.
        """
        return self.analyzer.label_centroid(
            label_name=label_name,
            load=load,
            centroid_face=centroid_face,
            custom_vertexes=custom_vertexes
        )
    
    def get_deepest_sulci(self, percentage: float = 10, label_name: Optional[str] = None, 
                          return_mask: bool = False, load_label: bool = False, 
                          result_label_name: Optional[str] = None):
        """
        Returns indices or a mask of vertices corresponding to the specified percentage 
        of deepest sulci.
        
        Parameters:
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
            numpy.ndarray: Indices or mask of deepest sulci
        """
        return self.analyzer.get_deepest_sulci(
            percentage=percentage,
            label_name=label_name,
            return_mask=return_mask,
            load_label=load_label,
            result_label_name=result_label_name
        )
    
    def threshold_label(self, label_name: str, threshold_type: str = 'absolute', 
                   threshold_direction: str = '>=', threshold_value: float = 0, 
                   threshold_measure: str = 'sulc', load_label: bool = False, 
                   new_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Threshold a label or the entire cortex based on a statistical value.

        Parameters:
            label_name (str): Name of the label, or 'cortex' to use the entire brain.
            threshold_type (str): Type of threshold - 'absolute' or 'percentile'.
            direction (str): Direction of thresholding - '>', '>=', '<', or '<='.
            threshold_value (float): Value to threshold at (absolute value or percentile).
            threshold_measure (str): Measure to threshold on - 'sulc', 'thickness', 'curv', or 'label_stat'.
            load_label (bool): If True, load the thresholded result as a new label.
            new_name (str): Name for the thresholded label if load_label is True.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Vertex indexes, RAS coordinates, and 
            measurement values of the thresholded result.
        """
        return self.analyzer.threshold_label(
            label_name=label_name,
            threshold_type=threshold_type,
            threshold_direction=threshold_direction,
            threshold_value=threshold_value,
            threshold_measure=threshold_measure,
            load_label=load_label,
            new_name=new_name
        )
    
    ############################
    # Measurement Methods 
    ############################
    
    def calculate_sulcal_depth(self, label_name: str, depth_pct: float = 8, 
                             n_deepest: int = 100, use_n_deepest: bool = True) -> float:
        """
        Calculate the depth of a sulcus.
        
        Parameters:
            label_name: str
                Name of the label corresponding to the sulcus
            depth_pct: float
                Percentage of deepest vertices to use (default: 8, matching MATLAB default)
            n_deepest: int
                Number of deepest vertices to use (default: 100)
            use_n_deepest: bool
                If True, use n_deepest vertices; if False, use depth_pct percentage (default: True)
                
        Returns:
            float: The median depth of the sulcus in mm
        """
        return self.measurer.calculate_sulcal_depth(
            label_name=label_name,
            depth_pct=depth_pct,
            n_deepest=n_deepest,
            use_n_deepest=use_n_deepest
        )
    
    def calculate_surface_area(self, label_name: Optional[str] = None) -> float:
        """
        Calculate the surface area of a label or the entire cortical surface.
        Replicates FreeSurfer's surface area calculation from mris_anatomical_stats
        
        Parameters:
            label_name: Optional[str]
                Name of the label to calculate area for. If None, calculates for the entire cortex.
                
        Returns:
            float: The surface area in mm²
        """
        return self.measurer.calculate_surface_area(label_name=label_name)
    
    def calculate_gray_matter_volume(self, label_name: Optional[str] = None) -> float:
        """
        Calculate gray matter volume between white and pial surfaces.
        Replicates FreeSurfer's volume calculation from mris_anatomical_stats
        
        Parameters:
            label_name: Optional[str]
                Name of the label to calculate volume for. If None, calculates for the entire cortex.
                
        Returns:
            float: The gray matter volume in mm³
        """
        return self.measurer.calculate_gray_matter_volume(label_name=label_name)
    
    def calculate_cortical_thickness(self, label_name: Optional[str] = None) -> Tuple[float, float]:
        """
        Calculate the mean and standard deviation of cortical thickness.
        Replicates FreeSurfer's thickness calculation from mris_anatomical_stats
        
        Parameters:
            label_name: Optional[str]
                Name of the label to calculate thickness for. If None, calculates for the entire cortex.
                
        Returns:
            Tuple[float, float]: Mean cortical thickness and standard deviation in mm
        """
        return self.measurer.calculate_cortical_thickness(label_name=label_name)
    
    def calculate_absolute_curvature(self, label_name: Optional[str] = None, 
                                   curvature_type: str = 'mean') -> float:
        """
        Calculate integrated rectified (absolute) curvature.
        Replicates FreeSurfer's MRIScomputeAbsoluteCurvature function.
        
        Parameters:
            label_name: Optional[str]
                Name of the label to calculate curvature for. If None, calculates for the entire cortex.
            curvature_type: str
                Type of curvature ('mean' or 'gaussian')
                
        Returns:
            float: Integrated rectified curvature
        """
        return self.measurer.calculate_absolute_curvature(
            label_name=label_name,
            curvature_type=curvature_type
        )
    
    def calculate_curvature_indices(self, label_name: Optional[str] = None) -> Tuple[float, float]:
        """
        Calculate folding index and intrinsic curvature index.
        Replicates FreeSurfer's MRIScomputeCurvatureIndices function.
        
        Parameters:
            label_name: Optional[str]
                Name of the label to calculate indices for. If None, calculates for the entire cortex.
                
        Returns:
            Tuple[float, float]: Folding index and intrinsic curvature index
        """
        return self.measurer.calculate_curvature_indices(label_name=label_name)
    
    def calculate_all_freesurfer_stats(self, label_name: str) -> Dict[str, float]:
        """
        Calculate all FreeSurfer anatomical statistics for a label.
        Replicates the complete output of mris_anatomical_stats
        
        Parameters:
            label_name: str
                Name of the label to calculate statistics for
                
        Returns:
            Dict[str, float]: Dictionary containing all anatomical measurements
        """
        return self.measurer.calculate_all_freesurfer_stats(label_name=label_name)
    
    def calculate_euclidean_distance(self, label1: str, label2: str, method: str = 'centroid') -> float:
        """
        Calculate the Euclidean distance between two labels.
        
        Parameters:
            label1: str
                Name of the first label
            label2: str
                Name of the second label
            method: str
                Method to use for calculating distance ('centroid', 'nearest', 'farthest')
                
        Returns:
            float: The Euclidean distance in mm
        """
        return self.measurer.calculate_euclidean_distance(
            label1=label1,
            label2=label2,
            method=method
        )
    
    def calculate_label_overlap(self, label1: str, label2: str) -> Dict[str, float]:
        """
        Calculate the overlap between two labels.
        
        Parameters:
            label1: str
                Name of the first label
            label2: str
                Name of the second label
                
        Returns:
            Dict[str, float]: A dictionary containing overlap metrics
        """
        return self.measurer.calculate_label_overlap(
            label1=label1,
            label2=label2
        )
    
    def export_measurements(self, labels: List[str], measurements: List[str], 
                          output_file: str, delimiter: str = ',') -> bool:
        """
        Export measurements for multiple labels to a file.
        
        Parameters:
            labels: List[str]
                List of label names to measure
            measurements: List[str]
                List of measurements to calculate:
                - 'area': Surface area
                - 'thickness': Cortical thickness  
                - 'depth': Sulcal depth
                - 'volume': Gray matter volume
                - 'curvature': Mean and Gaussian curvature
                - 'indices': Folding and intrinsic curvature indices
                - 'all_freesurfer': All FreeSurfer stats
            output_file: str
                Path to the output file
            delimiter: str
                Delimiter for the output file (default: ',')
                
        Returns:
            bool: True if successful, False otherwise
        """
        return self.measurer.export_measurements(
            labels=labels,
            measurements=measurements,
            output_file=output_file,
            delimiter=delimiter
        )