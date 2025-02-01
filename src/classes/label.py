from functools import cached_property
import numpy as np
import src.utilities.freesurfer_utils as sfu
from src.classes.subject import ScalpelSubject
import trimesh as tm


class Label(object):
    """
    Class for representing a freesurfer label.
    
    """
    
    def __init__(self, name, subject,  hemi, vertex_indexes=None, ras_coords=None, stat=None, custom_label_path=None):
        """
        Constructor for the Label class.

        Parameters:
        - name (str): Name of the label.
        - subject (ScalpelSubject): Subject object.
        - hemi (str): Hemisphere of the label.
        - from_file (bool, optional): If True, load label from file. Defaults to True.
        - vertex_indexes (np.array, optional): Numpy array of vertex indexes. Defaults to None.
        - ras_coords (np.array, optional): Numpy array of RAS coordinates. Defaults to None.
        - stat (np.array, optional): Numpy array of statistical values. Defaults to None.

        """
        if subject.hemi != hemi:
            raise Exception(f'The loaded subject hemisphere is {subject.hemi}, please specify the correct hemisphere `lh` or `rh` for this label')
        
        self._name = name
        self._subject = subject
        self._hemi = hemi

        if custom_label_path:
            self._vertex_indexes, self._ras_coords, self._stat = sfu.read_label(custom_label_path, include_stat = True)
        else:
            if vertex_indexes is None or ras_coords is None:
                raise Exception('Please provide vertex indexes and RAS coordinates for the label')
            self._vertex_indexes = vertex_indexes
            self._ras_coords = ras_coords
            self._stat = stat
            
    @property
    def name(self) -> str:
        return self._name
    
    
    @property
    def subject(self) -> ScalpelSubject:
        return self._subject
    
    @property
    def hemi(self) -> str:
        return self._subject.hemi
    
    @property
    def vertex_indexes(self) -> np.array:
        return self._vertex_indexes
    
    @property
    def ras_coords(self) -> np.array:
        return self._ras_coords

    def label_stat(self) -> np.array:
        return self._stat
    
    @property
    def faces(self) -> np.array:
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
        return self._subject.curv
    
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
            if self._subject.curv[point] < curv_threshold:
                gyrus_index.append(point)
                gyrus_RAS.append(RAS)

        # return np.array(gyrus_index), np.array(gyrus_RAS)              

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
            if self._subject.curv[point] > curv_threshold:
                sulcus_index.append(point)
                sulcus_RAS.append(RAS)

        return np.array(sulcus_index), np.array(sulcus_RAS)
    
    