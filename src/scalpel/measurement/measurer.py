from __future__ import annotations
from typing import List, Tuple, Dict, Union, TYPE_CHECKING, Optional
import numpy as np
from pathlib import Path
import csv

# Import utilities
from scalpel.utils import surface_utils

if TYPE_CHECKING:
    from scalpel.subject import ScalpelSubject

class ScalpelMeasurer:
    """
    Class for measuring brain surface data.
    
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

    
    def calculate_surface_area(self, label_name: Optional[str] = None) -> float:
        """
        Calculate the surface area of a label or the entire cortical surface.
        
        Parameters:
        -----------
        label_name: Optional[str]
            Name of the label to calculate area for. If None, calculates for the entire cortex.
            
        Returns:
        --------
        float
            The surface area in mm²
        """
        # Get the surface data
        
        return self.subject.labels[label_name].measurements['total surface area (mm^2)'] 
    
    def calculate_cortical_thickness(self, label_name: Optional[str] = None) -> float:
        """
        Calculate the mean cortical thickness of a label or the entire cortical surface.
        
        Parameters:
        -----------
        label_name: Optional[str]
            Name of the label to calculate thickness for. If None, calculates for the entire cortex.
            
        Returns:
        --------
        float
            The mean cortical thickness in mm
        """
        # Get thickness values from the subject's analyzer
        thickness_vals = self._subject.analyzer.thickness
        
        if label_name is not None:
            # Calculate thickness for a specific label
            if label_name not in self._subject.labels:
                raise ValueError(f"Label '{label_name}' not found in subject")
            
            # Get vertex indices for the label
            label_vertices = self._subject.labels[label_name].vertex_indexes
            
            # Get thickness values for the label vertices
            label_thickness = thickness_vals[label_vertices]
            
            # Calculate mean thickness
            mean_thickness = np.mean(label_thickness)
        else:
            # Calculate mean thickness for the entire cortical surface
            mean_thickness = np.mean(thickness_vals)
        
        return mean_thickness
    
    
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
        valid_measurements = ['area', 'thickness', 'depth']
        for measurement in measurements:
            if measurement not in valid_measurements:
                raise ValueError(f"Invalid measurement '{measurement}'. Choose from {valid_measurements}")
        
        # Prepare header
        header = ['label'] + measurements
        
        # Prepare data rows
        rows = []
        for label in labels:
            row = [label]
            
            for measurement in measurements:
                if measurement == 'area':
                    value = self.calculate_surface_area(label)
                elif measurement == 'thickness':
                    value = self.calculate_cortical_thickness(label)
                elif measurement == 'depth':
                    value = self.calculate_sulcal_depth(label)
                
                row.append(value)
            
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