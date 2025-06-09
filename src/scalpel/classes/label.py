from functools import cached_property
import numpy as np
import scalpel.utils.freesurfer_utils as fsu
import trimesh as tm
from pathlib import Path
from typing import Dict, Any, Optional, Union


class Label(object):
    """
    Class for representing a freesurfer label.
    """
    
    def __init__(self, name, hemi, subject_id=None, subjects_dir=None, vertex_indexes=None, 
                 ras_coords=None, stat=None, custom_label_path=None):
        """
        Constructor for the Label class.

        Parameters:
        - name (str): Name of the label.
        - hemi (str): Hemisphere of the label.
        - subject_id (str, optional): FreeSurfer subject ID. Defaults to None.
        - subjects_dir (str, optional): Path to FreeSurfer subjects directory. Defaults to None.
        - vertex_indexes (np.array, optional): Numpy array of vertex indexes. Defaults to None.
        - ras_coords (np.array, optional): Numpy array of RAS coordinates. Defaults to None.
        - stat (np.array, optional): Numpy array of statistical values. Defaults to None.
        - custom_label_path (str, optional): Path to a custom label file. Defaults to None.
        """
        self._label_name = name
        self._hemi = hemi
        self._subject_id = subject_id
        self._subjects_dir = subjects_dir
        self._label_stats = None  # Will be lazily loaded when needed

        if custom_label_path:
            self._vertex_indexes, self._ras_coords, self._stat = fsu.read_label(custom_label_path, include_stat=True)
        else:
            if vertex_indexes is None or ras_coords is None:
                raise Exception('Please provide vertex indexes and RAS coordinates for the label')
            self._vertex_indexes = vertex_indexes
            self._ras_coords = ras_coords
            self._stat = stat
            
    @property
    def label_name(self) -> str:
        return self._label_name
    
    @property
    def hemi(self) -> str:
        return self._subject.hemi if hasattr(self, '_subject') else self._hemi

    @property
    def subject(self) -> object:
        """
        Returns the subject object associated with this label.

        Returns:
        - ScalpelSubject: The subject object.
        """
        from scalpel.subject import ScalpelSubject
        if not hasattr(self, '_subject'):
            self._subject = ScalpelSubject(subject_id=self._subject_id, subjects_dir=self._subjects_dir, hemi=self._hemi)
        return self._subject
    
    @property
    def vertex_indexes(self) -> np.array:
        return self._vertex_indexes
    
    @property
    def label_RAS(self) -> np.array:
        return self._ras_coords

    @property
    def label_stat(self) -> np.array:
        return self._stat
    
    @property
    def measurements(self) -> Dict[str, Any]:
        """
        Get the measurements for this label by loading the LabelStats if not already loaded.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing label statistics measurements
        """
        if self._label_stats is None:
            self.load_stats()
        
        return self._label_stats.measurements if self._label_stats else {}
    
    @property
    def faces(self) -> np.array:
        if not hasattr(self, '_subject'):
            self._subject = self.subject
            
        label_faces = []
        for face in self._subject.faces:
            for vertex_index in face:
                if vertex_index in self._vertex_indexes:
                    label_faces.append(face)
                    
        return np.array(label_faces)
    
    @cached_property
    def mesh(self) -> tm.Trimesh:
        return tm.Trimesh(vertices=self.subject.ras_coords, faces=self.faces)
    
    @cached_property
    def curv(self) -> np.array:
        return self.subject.curv
    
    def gyrus(self, curv_threshold=0) -> np.array:
        """
        Returns all label indices and RAS coordinates for gyrus within the freesurfer label.

        Parameters:
        - curv_threshold (int, optional): Value for thresholding curvature value. Defaults to 0.

        Returns:
        - np.array: Numpy array of gyrus indexes.
        - np.array: Numpy array of gyrus RAS vertices.
        """
        gyrus_index = []
        gyrus_RAS = []

        for point, RAS in zip(self._vertex_indexes, self._ras_coords):
            if self.curv[point] < curv_threshold:
                gyrus_index.append(point)
                gyrus_RAS.append(RAS)

        return np.array(gyrus_index), np.array(gyrus_RAS)              

    def sulcus(self, curv_threshold=0) -> np.array:
        """
        Returns all label indices and RAS coordinates for sulcus within the freesurfer label.

        Parameters:
        - curv_threshold (int, optional): Value for thresholding curvature value. Defaults to 0.

        Returns:
        - np.array: Numpy array of sulcus indexes.
        - np.array: Numpy array of sulcus RAS vertices.
        """
        sulcus_index = []
        sulcus_RAS = []

        for point, RAS in zip(self._vertex_indexes, self._ras_coords):
            if self.curv[point] > curv_threshold:
                sulcus_index.append(point)
                sulcus_RAS.append(RAS)

        return np.array(sulcus_index), np.array(sulcus_RAS)
    
    def write_label(self, label_name, label_dir_path=None, overwrite=False):
        """
        Writes label to file.

        Parameters:
        - label_name (str): Name for the output label file.
        - label_dir_path (str, optional): Directory path to save the label. Defaults to None.
        - overwrite (bool, optional): Whether to overwrite existing file. Defaults to False.
        """
        if label_dir_path is None:
            label_dir_path = self._subjects_dir / self._subject_id / 'label' 
            
        fsu.write_label(
            label_name=label_name, 
            label_indexes=self._vertex_indexes, 
            label_RAS=self._ras_coords, 
            hemi=self.subject.hemi,
            subjects_dir=self.subject.subjects_dir,
            subject_id=self.subject.subject_id,
            surface_type=self.subject.surface_type,
            custom_label_dir=label_dir_path, 
            overwrite=overwrite
        )
    
    def load_stats(self, stats_filepath: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load statistics for this label from a FreeSurfer stats file using the LabelStats class.
        
        Parameters:
        -----------
        stats_filepath : Optional[Union[str, Path]]
            Path to the FreeSurfer statistics file. If None, it will try to use a default path.
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing the parsed statistics
        """
        from scalpel.classes.label_stats import LabelStats
        
        self._label_stats = LabelStats(self, stats_filepath)
        return self._label_stats.measurements
    
    def get_measurement(self, key: str) -> Any:
        """
        Get a specific measurement value.
        
        Parameters:
        -----------
        key : str
            The measurement key to retrieve
            
        Returns:
        --------
        Any
            The value of the measurement, or None if not found
        """
        if self._label_stats is None:
            self.load_stats()
        
        return self._label_stats.get_measurement(key) if self._label_stats else None