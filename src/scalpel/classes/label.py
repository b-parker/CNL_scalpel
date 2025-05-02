from functools import cached_property
import numpy as np
import scalpel.utilities.freesurfer_utils as sfu
import trimesh as tm


class Label(object):
    """
    Class for representing a freesurfer label.
    
    """
    
    def __init__(self, name, hemi, subject_id = None, subjects_dir = None, vertex_indexes=None, ras_coords=None, stat=None, custom_label_path=None):
        """
        Constructor for the Label class.

        Parameters:
        - name (str): Name of the label.
        - hemi (str): Hemisphere of the label.
        - from_file (bool, optional): If True, load label from file. Defaults to True.
        - vertex_indexes (np.array, optional): Numpy array of vertex indexes. Defaults to None.
        - ras_coords (np.array, optional): Numpy array of RAS coordinates. Defaults to None.
        - stat (np.array, optional): Numpy array of statistical values. Defaults to None.

        """
        self._label_name = name
        self._hemi = hemi
        self._subject_id = subject_id
        self._subjects_dir = subjects_dir

        if custom_label_path:
            self._vertex_indexes, self._ras_coords, self._stat = sfu.read_label(custom_label_path, include_stat = True)
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
        return self._subject.hemi

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
    def ras_coords(self) -> np.array:
        return self._ras_coords

    @property
    def label_stat(self) -> np.array:
        return self._stat
    
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
        Returns all label indices and RAS coordinates for gyrus within the freesurfer label.

        Parameters:
        - curv_threshold (int, optional): Value for thresholding curvature value. Defaults to 0.

        Returns:
        - np.array: Numpy array of gyrus indexes.
        - np.array: Numpy array of gyrus RAS vertices.
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
        - custom_label_path (str): Path to save the label.

        """
        if label_dir_path is None:
            label_dir_path = self._subjects_dir / self._subject_id / 'label' 
            
        sfu.write_label(
            label_name = label_name, 
            label_indexes = self._vertex_indexes, 
            label_RAS = self._ras_coords, 
            hemi = self.subject.hemi,
            subjects_dir = self.subject.subjects_dir,
            subject_id= self.subject.subject_id,
            surface_type= self.subject.surface_type,
            custom_label_dir = label_dir_path, 
            overwrite=overwrite
            )
    
    