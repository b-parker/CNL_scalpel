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
        Calculate the depth of a sulcus matching the MATLAB calcSulc_depth function.
        
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

        NOTE: Requires the pial and gyral-inflated surfaces to be generated with recon-all -all
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
                
            # Calculate distances from pial to gyral-inflated surface
            depths = []
            for vertex_idx in fundus_vertices:
                # Get coordinates of the vertex on the pial surface
                v_xyz = self.subject.pial_v[vertex_idx]
                    
                # Calculate distances to all gyral-inflated vertices
                # NOTE: The gyral-inflated surface is generated with recon-all flag -all
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
        white_vertices = self.subject.white_v
        pial_vertices = self.subject.pial_v
        faces = self.subject.faces
        thickness_vals = self.subject.thickness  # or self.subject.analyzer.thickness
        
        total_volume = 0.0
        
        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            label_vertex_indices = self.subject.labels[label_name].vertex_indexes
            label_vertex_set = set(label_vertex_indices)
        
        # Process each face
        for face in faces:
            # Check if face belongs to the label (if specified)
            if label_name is not None:
                face_in_label = any(v_idx in label_vertex_set for v_idx in face)
                if not face_in_label:
                    continue
            
            # Calculate average thickness for this face
            face_thickness = np.mean([thickness_vals[v_idx] for v_idx in face])
            
            # Calculate white surface face area
            white_face_coords = white_vertices[face]
            white_face_area = self._get_face_area(white_face_coords)
            
            # Calculate pial surface face area
            pial_face_coords = pial_vertices[face]
            pial_face_area = self._get_face_area(pial_face_coords)
            
            # Volume is average thickness * average of white and pial areas
            # This matches the FreeSurfer calculation: volume = avg_thick * (white_area + pial_area) / 2
            face_volume = face_thickness * (white_face_area + pial_face_area) / 2.0
            
            if label_name is not None:
                # Distribute volume to vertices in the label
                for v_idx in face:
                    if v_idx in label_vertex_set:
                        total_volume += face_volume / 3.0  # Each vertex gets 1/3
            else:
                total_volume += face_volume
        
        # Divide by 2 to match FreeSurfer's final volume calculation
        total_volume /= 2.0
        
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
            
            # Calculate weighted absolute curvature
            total_weighted_curvature = np.sum(np.abs(label_curvature) * label_areas)
            total_area = np.sum(label_areas)
            
            integrated_curvature = total_weighted_curvature / len(label_vertices) if len(label_vertices) > 0 else 0.0
            
            # Store in measurements
            if curvature_type == 'mean':
                self.subject.labels[label_name].measurements['integrated rectified mean curvature'] = integrated_curvature
            else:
                self.subject.labels[label_name].measurements['integrated rectified gaussian curvature'] = integrated_curvature
        else:
            # Calculate for entire surface
            total_weighted_curvature = np.sum(np.abs(curvature_vals) * vertex_areas)
            integrated_curvature = total_weighted_curvature / len(curvature_vals)
        
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
        mean_curvature = self.subject.mean_curvature
        gaussian_curvature = self.subject.gaussian_curvature
        vertices = self.subject.white_v
        faces = self.subject.faces
        
        # Calculate vertex areas
        vertex_areas = self._compute_vertex_areas(vertices, faces)
        
        if label_name is not None:
            if label_name not in self.subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            label_vertices = self.subject.labels[label_name].vertex_indexes
            label_mean_curv = mean_curvature[label_vertices]
            label_gauss_curv = gaussian_curvature[label_vertices]
            label_areas = vertex_areas[label_vertices]
        else:
            label_mean_curv = mean_curvature
            label_gauss_curv = gaussian_curvature
            label_areas = vertex_areas
        
        # Folding Index: sum of |mean_curvature| * area where mean_curvature > 0
        positive_mean_mask = label_mean_curv > 0
        folding_index = np.sum(np.abs(label_mean_curv[positive_mean_mask]) * label_areas[positive_mean_mask])
        
        # Intrinsic Curvature Index: sum of |gaussian_curvature| * area where gaussian_curvature > 0
        positive_gauss_mask = label_gauss_curv > 0
        intrinsic_curvature_index = np.sum(np.abs(label_gauss_curv[positive_gauss_mask]) * label_areas[positive_gauss_mask])
        
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
        
        # Store all results in the label's measurements
        for key, value in results.items():
            self.subject.labels[label_name].measurements[key] = value
        
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
        coords1 = self._subject.labels[label1].ras_coords
        coords2 = self._subject.labels[label2].ras_coords
        
        if method == 'centroid':
            # Calculate centroid for each label
            centroid1 = np.mean(coords1, axis=0)
            centroid2 = np.mean(coords2, axis=0)
            
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