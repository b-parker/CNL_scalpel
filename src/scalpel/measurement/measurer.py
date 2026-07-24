from __future__ import annotations
from typing import List, Tuple, Dict, Union, TYPE_CHECKING, Optional
import numpy as np
from pathlib import Path
import csv


from scalpel.utils import surface_utils

if TYPE_CHECKING:
    from scalpel.subject import ScalpelSubject

class ScalpelMeasurer:
    """
    Class for measuring surface data.
    
    This class provides measurement functionality for a ScalpelSubject,
    including calculation of sulcal depth, surface area, cortical thickness,
    distances between labels, and overlap between labels.
    """
    
    def __init__(self, subject: 'ScalpelSubject'):
        """
        Initialize a ScalpelMeasurer.
        
        Parameters:
        -----------
        subject : ScalpelSubject
            The subject to measure
        """
        self._subject = subject
    
    @property
    def subject(self):
        """Get the associated ScalpelSubject."""
        return self._subject
    

    def calculate_sulcal_depth(self, label_name, depth_pct=8, n_deepest=100, use_n_deepest=True):
        """
        Sulcal depth following Madan (2019) calcSulc: the median distance from the
        deepest sulcal (pial) vertices to the nearest point on the
        ``pial-outer-smoothed`` gyral hull.

        Reference: Madan (2019), Brain Informatics 6:5.
        https://doi.org/10.1186/s40708-019-0098-1 -- https://github.com/cMadan/calcSulc

        Parameters:
        -----------
        label_name: str
            Name of the label corresponding to the sulcus
        depth_pct: float
            Percentage of deepest vertices to use (default: 8)
        n_deepest: int
            Number of deepest vertices to use (default: 100)
        use_n_deepest: bool
            If True, use n_deepest vertices; if False, use depth_pct percentage (default: True)
                
        Returns:
        --------
        float: The median depth of the sulcus in mm

        NOTE: Requires the pial and pial-outer-smoothed surfaces (recon-all -localGI).
        """
        try:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found")
                    
            label_vertices = self.subject.labels[label_name].vertex_indexes
                
            if not isinstance(label_vertices, np.ndarray):
                label_vertices = np.array(label_vertices, dtype=int)
                
            sulc_map = self.subject.sulc_vals
            
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
                
            depths = []
            for vertex_idx in fundus_vertices:
                # min distance from this pial fundus vertex to the gyral hull
                v_xyz = self.subject.pial_v[vertex_idx]
                distances = np.sqrt(np.sum((self.subject.gyrif_v - v_xyz)**2, axis=1))
                    
                # Find minimum distance
                min_distance = np.min(distances)
                depths.append(min_distance)
            
            self.subject.labels[label_name].measurements['sulcal depth (mm)'] = np.median(depths)
                
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

    def _get_face_area(self, face_vertices):
        """
        Calculate the area of a triangular face using cross product.
        Replicates the FreeSurfer face area calculation.
        
        Parameters:
        -----------
        face_vertices: np.ndarray
            3x3 array of vertex coordinates for the face
            
        Returns:
        --------
        float
            Area of the face
        """
        v0, v1, v2 = face_vertices
        # Calculate cross product of two edge vectors
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        # Area is half the magnitude of cross product
        return 0.5 * np.linalg.norm(cross)

    def _compute_vertex_areas(self, vertices, faces, label_vertices=None):
        """
        Compute area associated with each vertex (1/3 of adjacent face areas).
        Matches FreeSurfer's vertex area calculation.
        
        Parameters:
        -----------
        vertices: np.ndarray
            Vertex coordinates
        faces: np.ndarray
            Face connectivity
        label_vertices: np.ndarray, optional
            Specific vertices to compute areas for
            
        Returns:
        --------
        np.ndarray
            Area associated with each vertex
        """
        vertex_areas = np.zeros(len(vertices))
        
        for face in faces:
            # Get face vertices
            face_coords = vertices[face]
            face_area = self._get_face_area(face_coords)
            
            # Each vertex gets 1/3 of the face area (VERTICES_PER_FACE = 3)
            for vertex_idx in face:
                vertex_areas[vertex_idx] += face_area / 3.0
        
        if label_vertices is not None:
            return vertex_areas[label_vertices]
        
        return vertex_areas

    def calculate_surface_area(self, label_name: Optional[str] = None) -> float:
        """
        Calculate the surface area of a label or the entire cortical surface.
        Replicates FreeSurfer's surface area calculation from mris_anatomical_stats
        
        Parameters:
        -----------
        label_name: Optional[str]
            Name of the label to calculate area for. If None, calculates for the entire cortex.
            
        Returns:
        --------
        float
            The surface area in mm²
        """
        # Use the original surface (white matter surface typically)
        vertices = self.subject.white_v  # or whatever surface coordinates are available
        faces = self.subject.faces
        
        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            # Get vertex indices for the label
            label_vertices = self.subject.labels[label_name].vertex_indexes
            
            # Calculate vertex areas for the label
            vertex_areas = self._compute_vertex_areas(vertices, faces, label_vertices)
            total_area = np.sum(vertex_areas)
            
            # Store in measurements
            self.subject.labels[label_name].measurements['total surface area (mm^2)'] = total_area
        else:
            # Calculate total surface area
            vertex_areas = self._compute_vertex_areas(vertices, faces)
            total_area = np.sum(vertex_areas)
        
        return total_area

    def calculate_gray_matter_volume(self, label_name: Optional[str] = None) -> float:
        """
        Calculate gray matter volume between white and pial surfaces.
        Replicates FreeSurfer's volume calculation from mris_anatomical_stats
        
        Parameters:
        -----------
        label_name: Optional[str]
            Name of the label to calculate volume for. If None, calculates for the entire cortex.
            
        Returns:
        --------
        float
            The gray matter volume in mm³
        """
        white = self.subject.white_v
        pial = self.subject.pial_v
        faces = self.subject.faces

        # Per-face volume (FreeSurfer MRISvolumeTH3): each white->pial prism split
        # into 3 tetrahedra, summed as |scalar triple product| / 6.
        Pa, Pb, Pc = pial[faces[:, 0]], pial[faces[:, 1]], pial[faces[:, 2]]
        Wa, Wb, Wc = white[faces[:, 0]], white[faces[:, 1]], white[faces[:, 2]]
        Bp, Cp = Pb - Pa, Pc - Pa
        Aw, Bw, Cw = Wa - Pa, Wb - Pa, Wc - Pa
        T1 = np.abs(np.einsum('ij,ij->i', Aw, np.cross(Bw, Cw)))
        T2 = np.abs(np.einsum('ij,ij->i', Bp, np.cross(Cp, Bw)))
        T3 = np.abs(np.einsum('ij,ij->i', Cp, np.cross(Cw, Bw)))
        face_volumes = (T1 + T2 + T3) / 6.0

        # 1/3 of each face's volume to each vertex (FreeSurfer "vertex volume").
        vertex_volumes = np.zeros(len(white))
        for col in range(3):
            np.add.at(vertex_volumes, faces[:, col], face_volumes / 3.0)

        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            target_vertices = self.subject.labels[label_name].vertex_indexes
        else:
            # FreeSurfer masks the whole-hemisphere volume to cortex (no medial wall).
            target_vertices = self.subject.cortex_vertices

        total_volume = float(np.sum(vertex_volumes[target_vertices]))

        if label_name is not None:
            self.subject.labels[label_name].measurements['gray matter volume (mm^3)'] = total_volume

        return total_volume

    def calculate_cortical_thickness(self, label_name: Optional[str] = None) -> Tuple[float, float]:
        """
        Calculate the mean and standard deviation of cortical thickness.
        Replicates FreeSurfer's thickness calculation from mris_anatomical_stats
        
        Parameters:
        -----------
        label_name: Optional[str]
            Name of the label to calculate thickness for. If None, calculates for the entire cortex.
            
        Returns:
        --------
        Tuple[float, float]
            Mean cortical thickness and standard deviation in mm
        """
        thickness_vals = self.subject.thickness  # or self.subject.analyzer.thickness
        
        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            # Get vertex indices for the label
            label_vertices = self.subject.labels[label_name].vertex_indexes
            
            # Get thickness values for the label vertices
            label_thickness = thickness_vals[label_vertices]
            
            # Calculate mean and standard deviation
            mean_thickness = np.mean(label_thickness)
            std_thickness = np.std(label_thickness, ddof=0)  # Population std, like FreeSurfer
            
            # Store in measurements
            self.subject.labels[label_name].measurements['average cortical thickness (mm)'] = mean_thickness
            self.subject.labels[label_name].measurements['cortical thickness std (mm)'] = std_thickness
        else:
            # Calculate for entire cortical surface
            mean_thickness = np.mean(thickness_vals)
            std_thickness = np.std(thickness_vals, ddof=0)
        
        return mean_thickness, std_thickness

    def calculate_absolute_curvature(self, label_name: Optional[str] = None, curvature_type: str = 'mean') -> float:
        """
        Calculate integrated rectified (absolute) curvature.
        Replicates FreeSurfer's MRIScomputeAbsoluteCurvature function.
        
        Parameters:
        -----------
        label_name: Optional[str]
            Name of the label to calculate curvature for. If None, calculates for the entire cortex.
        curvature_type: str
            Type of curvature ('mean' or 'gaussian')
            
        Returns:
        --------
        float
            Integrated rectified curvature
        """
        if curvature_type == 'mean':
            curvature_vals = self.subject.mean_curvature  # or appropriate curvature data
        elif curvature_type == 'gaussian':
            curvature_vals = self.subject.gaussian_curvature
        else:
            raise ValueError("curvature_type must be 'mean' or 'gaussian'")
        
        vertices = self.subject.white_v  # Use white surface coordinates
        faces = self.subject.faces
        
        # Calculate vertex areas
        vertex_areas = self._compute_vertex_areas(vertices, faces)
        
        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            label_vertices = self.subject.labels[label_name].vertex_indexes
            label_curvature = curvature_vals[label_vertices]
            label_areas = vertex_areas[label_vertices]
            
            # FreeSurfer's integrated rectified curvature: area-weighted |curvature|
            # summed over the label, divided by vertex count.
            integrated_curvature = (np.sum(np.abs(label_curvature) * label_areas) / len(label_vertices)
                                    if len(label_vertices) > 0 else 0.0)

            # Store in measurements
            if curvature_type == 'mean':
                self.subject.labels[label_name].measurements['integrated rectified mean curvature'] = integrated_curvature
            else:
                self.subject.labels[label_name].measurements['integrated rectified gaussian curvature'] = integrated_curvature
        else:
            # Calculate for entire surface
            integrated_curvature = np.sum(np.abs(curvature_vals) * vertex_areas) / len(curvature_vals)
        
        return integrated_curvature

    def calculate_curvature_indices(self, label_name: Optional[str] = None) -> Tuple[float, float]:
        """
        Calculate folding index and intrinsic curvature index.
        Replicates FreeSurfer's MRIScomputeCurvatureIndices function.
        
        Parameters:
        -----------
        label_name: Optional[str]
            Name of the label to calculate indices for. If None, calculates for the entire cortex.
            
        Returns:
        --------
        Tuple[float, float]
            Folding index and intrinsic curvature index
        """
        k1 = self.subject.k1_curvature
        k2 = self.subject.k2_curvature
        vertices = self.subject.white_v
        faces = self.subject.faces

        # Calculate vertex areas
        vertex_areas = self._compute_vertex_areas(vertices, faces)

        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")

            label_vertices = self.subject.labels[label_name].vertex_indexes
            k1 = k1[label_vertices]
            k2 = k2[label_vertices]
            label_areas = vertex_areas[label_vertices]
        else:
            label_areas = vertex_areas

        abs_k1, abs_k2 = np.abs(k1), np.abs(k2)

        # Folding index: (1/4pi) * integral |k1|*(|k1|-|k2|) dA
        folding_index = np.sum(abs_k1 * (abs_k1 - abs_k2) * label_areas) / (4.0 * np.pi)

        # Intrinsic curvature index: (1/4pi) * integral of positive Gaussian (k1*k2) dA
        gaussian = k1 * k2
        intrinsic_curvature_index = np.sum(np.maximum(gaussian, 0.0) * label_areas) / (4.0 * np.pi)
        
        if label_name is not None:
            self.subject.labels[label_name].measurements['folding index'] = folding_index
            self.subject.labels[label_name].measurements['intrinsic curvature index'] = intrinsic_curvature_index
        
        return folding_index, intrinsic_curvature_index

    def calculate_all_freesurfer_stats(self, label_name: str) -> Dict[str, float]:
        """
        Calculate all FreeSurfer anatomical statistics for a label.
        Replicates the complete output of mris_anatomical_stats
        
        Parameters:
        -----------
        label_name: str
            Name of the label to calculate statistics for
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing all anatomical measurements
        """
        if label_name not in self.subject.labels:
            raise ValueError(f"Label '{label_name}' not found in subject")
        
        results = {}
        
        # Number of vertices
        label_vertices = self.subject.labels[label_name].vertex_indexes
        results['num_vertices'] = len(label_vertices)
        
        # Surface area
        results['surface_area_mm2'] = self.calculate_surface_area(label_name)
        
        # Gray matter volume
        results['gray_volume_mm3'] = self.calculate_gray_matter_volume(label_name)
        
        # Cortical thickness
        mean_thick, std_thick = self.calculate_cortical_thickness(label_name)
        results['thickness_mean_mm'] = mean_thick
        results['thickness_std_mm'] = std_thick
        
        # Curvature measures
        results['mean_curvature'] = self.calculate_absolute_curvature(label_name, 'mean')
        results['gaussian_curvature'] = self.calculate_absolute_curvature(label_name, 'gaussian')
        
        # Curvature indices
        folding_idx, intrinsic_idx = self.calculate_curvature_indices(label_name)
        results['folding_index'] = folding_idx
        results['intrinsic_curvature_index'] = intrinsic_idx

        # The sub-methods above already cache their (descriptive) keys onto the
        # label; return the compact dict for programmatic use.
        return results
    
    def calculate_euclidean_distance(self, label1: str, label2: str, method: str = 'centroid') -> float:
        """
        Calculate the Euclidean distance between two labels in 3D space.
        
        Parameters:
        -----------
        label1: str
            Name of the first label
        label2: str
            Name of the second label
        method: str
            Method to use for calculating distance ('centroid', 'nearest', 'farthest')
            
        Returns:
        --------
        float
            The Euclidean distance in mm
        """
        # Check if labels exist
        if label1 not in self._subject.labels:
            raise ValueError(f"Label '{label1}' not found in subject")
        if label2 not in self._subject.labels:
            raise ValueError(f"Label '{label2}' not found in subject")
        
        # Get RAS coordinates for each label
        coords1 = self._subject.labels[label1].label_RAS
        coords2 = self._subject.labels[label2].label_RAS
        
        if method == 'centroid':
            # Area-weighted geometric centroid of each label (falls back to the
            # unweighted vertex mean for labels with no complete interior faces)
            centroid1 = self._label_centroid(label1)
            centroid2 = self._label_centroid(label2)

            # Calculate Euclidean distance between centroids
            distance = np.linalg.norm(centroid2 - centroid1)
        
        elif method == 'nearest':
            # Find pair of vertices from each label with minimum distance
            min_distance = float('inf')
            
            # Calculate pairwise distances between all vertices
            for c1 in coords1:
                for c2 in coords2:
                    dist = np.linalg.norm(c2 - c1)
                    if dist < min_distance:
                        min_distance = dist
            
            distance = min_distance
        
        elif method == 'farthest':
            # Find pair of vertices from each label with maximum distance
            max_distance = 0
            
            # Calculate pairwise distances between all vertices
            for c1 in coords1:
                for c2 in coords2:
                    dist = np.linalg.norm(c2 - c1)
                    if dist > max_distance:
                        max_distance = dist
            
            distance = max_distance
        
        else:
            raise ValueError("Invalid method. Choose 'centroid', 'nearest', or 'farthest'.")

        return distance

    def _label_centroid(self, label_name: str) -> np.ndarray:
        """Area-weighted geometric centroid of a label (vertex-mean fallback if no faces)."""
        label = self._subject.labels[label_name]
        label_faces = surface_utils.get_faces_from_vertices(
            self._subject.faces, label.vertex_indexes
        )
        if len(label_faces) == 0:
            return np.mean(label.label_RAS, axis=0)
        try:
            return surface_utils.calculate_geometric_centroid(
                self._subject.surface_RAS, label_faces
            )
        except ValueError:
            return np.mean(label.label_RAS, axis=0)

    def _get_geodesic_algorithm(self):
        """Lazily build and cache an exact geodesic solver over the loaded surface."""
        if getattr(self, '_geoalg', None) is None:
            try:
                import pygeodesic.geodesic as geodesic
            except ImportError as exc:
                raise ImportError(
                    "Geodesic distance requires the 'pygeodesic' package. "
                    "Install it with `pip install pygeodesic`."
                ) from exc
            vertices = np.asarray(self._subject.surface_RAS, dtype=np.float64)
            faces = np.asarray(self._subject.faces, dtype=np.int32)
            self._geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
        return self._geoalg

    def _nearest_label_vertex_to_centroid(self, label_name: str) -> int:
        """Return the label's vertex index that is closest to its centroid."""
        label = self._subject.labels[label_name]
        centroid = self._label_centroid(label_name)
        local_idx = np.argmin(np.linalg.norm(label.label_RAS - centroid, axis=1))
        return int(label.vertex_indexes[local_idx])

    def calculate_geodesic_distance(self, label1: str, label2: str, method: str = 'centroid') -> float:
        """
        Exact geodesic (on-surface) distance between two labels, measured across
        the subject's loaded surface via the MMP algorithm (`pygeodesic`).

        Parameters:
        -----------
        label1: str
            Name of the first label
        label2: str
            Name of the second label
        method: str
            'centroid' - geodesic distance between each label's centroid vertex
            'nearest'  - minimum geodesic distance between the two label vertex sets

        Returns:
        --------
        float
            The geodesic distance in mm
        """
        if label1 not in self._subject.labels:
            raise ValueError(f"Label '{label1}' not found in subject")
        if label2 not in self._subject.labels:
            raise ValueError(f"Label '{label2}' not found in subject")

        geoalg = self._get_geodesic_algorithm()

        if method == 'centroid':
            source = self._nearest_label_vertex_to_centroid(label1)
            target = self._nearest_label_vertex_to_centroid(label2)
            distance, _ = geoalg.geodesicDistance(source, target)
            return float(distance)

        elif method == 'nearest':
            # min geodesic distance between the two label vertex sets (one wavefront)
            sources = np.asarray(
                self._subject.labels[label1].vertex_indexes, dtype=np.int32
            )
            targets = np.asarray(
                self._subject.labels[label2].vertex_indexes, dtype=np.int32
            )
            distances, _ = geoalg.geodesicDistances(sources, targets)
            return float(np.min(distances))

        else:
            raise ValueError("Invalid method. Choose 'centroid' or 'nearest'.")

    def calculate_label_overlap(self, label1: str, label2: str) -> Dict[str, float]:
        """
        Calculate the overlap between two labels using multiple metrics.
        
        Parameters:
        -----------
        label1: str
            Name of the first label
        label2: str
            Name of the second label
            
        Returns:
        --------
        Dict[str, float]
            A dictionary containing overlap metrics:
            - 'dice': Dice coefficient (2*|A∩B| / (|A|+|B|))
            - 'jaccard': Jaccard index (|A∩B| / |A∪B|)
            - 'overlap_coefficient': Overlap coefficient (|A∩B| / min(|A|,|B|))
            - 'intersection_size': Size of intersection (number of vertices)
            - 'union_size': Size of union (number of vertices)
        """
        # Check if labels exist
        if label1 not in self._subject.labels:
            raise ValueError(f"Label '{label1}' not found in subject")
        if label2 not in self._subject.labels:
            raise ValueError(f"Label '{label2}' not found in subject")
        
        # Get vertex indices for each label
        vertices1 = self._subject.labels[label1].vertex_indexes
        vertices2 = self._subject.labels[label2].vertex_indexes
        
        # Calculate intersection and union
        intersection = np.intersect1d(vertices1, vertices2)
        union = np.union1d(vertices1, vertices2)
        
        # Calculate sizes
        size1 = len(vertices1)
        size2 = len(vertices2)
        intersection_size = len(intersection)
        union_size = len(union)
        
        # Calculate overlap metrics
        dice = 2 * intersection_size / (size1 + size2) if (size1 + size2) > 0 else 0
        jaccard = intersection_size / union_size if union_size > 0 else 0
        overlap_coefficient = intersection_size / min(size1, size2) if min(size1, size2) > 0 else 0
        
        # Return metrics as a dictionary
        return {
            'dice': dice,
            'jaccard': jaccard,
            'overlap_coefficient': overlap_coefficient,
            'intersection_size': intersection_size,
            'union_size': union_size
        }
    
    def export_measurements(self, labels: List[str], measurements: List[str], 
                           output_file: str, delimiter: str = ',') -> bool:
        """
        Export measurements for multiple labels to a CSV file.
        
        Parameters:
        -----------
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
            Delimiter for the CSV file (default: ',')
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        # Check if labels exist
        for label in labels:
            if label not in self._subject.labels:
                raise ValueError(f"Label '{label}' not found in subject")
        
        # Check measurement types
        valid_measurements = ['area', 'thickness', 'depth', 'volume', 'curvature', 'indices', 'all_freesurfer']
        for measurement in measurements:
            if measurement not in valid_measurements:
                raise ValueError(f"Invalid measurement '{measurement}'. Choose from {valid_measurements}")
        
        # Prepare header based on requested measurements
        header = ['label']
        for measurement in measurements:
            if measurement == 'area':
                header.append('surface_area_mm2')
            elif measurement == 'thickness':
                header.extend(['thickness_mean_mm', 'thickness_std_mm'])
            elif measurement == 'depth':
                header.append('sulcal_depth_mm')
            elif measurement == 'volume':
                header.append('gray_volume_mm3')
            elif measurement == 'curvature':
                header.extend(['mean_curvature', 'gaussian_curvature'])
            elif measurement == 'indices':
                header.extend(['folding_index', 'intrinsic_curvature_index'])
            elif measurement == 'all_freesurfer':
                header.extend(['num_vertices', 'surface_area_mm2', 'gray_volume_mm3', 
                              'thickness_mean_mm', 'thickness_std_mm', 'mean_curvature', 
                              'gaussian_curvature', 'folding_index', 'intrinsic_curvature_index'])
        
        # Prepare data rows
        rows = []
        for label in labels:
            row = [label]
            
            for measurement in measurements:
                if measurement == 'area':
                    value = self.calculate_surface_area(label)
                    row.append(value)
                elif measurement == 'thickness':
                    mean_thick, std_thick = self.calculate_cortical_thickness(label)
                    row.extend([mean_thick, std_thick])
                elif measurement == 'depth':
                    value = self.calculate_sulcal_depth(label)
                    row.append(value)
                elif measurement == 'volume':
                    value = self.calculate_gray_matter_volume(label)
                    row.append(value)
                elif measurement == 'curvature':
                    mean_curv = self.calculate_absolute_curvature(label, 'mean')
                    gauss_curv = self.calculate_absolute_curvature(label, 'gaussian')
                    row.extend([mean_curv, gauss_curv])
                elif measurement == 'indices':
                    fold_idx, intrinsic_idx = self.calculate_curvature_indices(label)
                    row.extend([fold_idx, intrinsic_idx])
                elif measurement == 'all_freesurfer':
                    stats = self.calculate_all_freesurfer_stats(label)
                    row.extend([stats['num_vertices'], stats['surface_area_mm2'], 
                               stats['gray_volume_mm3'], stats['thickness_mean_mm'], 
                               stats['thickness_std_mm'], stats['mean_curvature'], 
                               stats['gaussian_curvature'], stats['folding_index'], 
                               stats['intrinsic_curvature_index']])
            
            rows.append(row)
        
        # Write to CSV file
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=delimiter)
                writer.writerow(header)
                writer.writerows(rows)
            
            return True
        
        except Exception as e:
            print(f"Error writing to CSV file: {e}")
            return False