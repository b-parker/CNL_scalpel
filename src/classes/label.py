from functools import cached_property
import numpy as np
import src.utilities.freesurfer_utils as sfu
import trimesh as tm


# TODO: move global variable setup somewhere else!
SUBJECTS_DIR = '/Users/megananctil/WeinerLab/subjects'

class Label(object):
    
    def __init__(self, name, subject, from_file=True, hemi="lh", vertex_indexes=None, ras_coords=None):
        # (MA) Not sure if we even want to feed in hemi for label creation if the subject
        # fed in will just restrict it anyway
        self._subject_dir = f'{SUBJECTS_DIR}/{subject.name}'
        if subject.hemi != hemi:
            raise Exception(f'The loaded subject hemisphere is {subject.hemi}, please specify the correct hemisphere `lh` or `rh` for this label')
        
        self._name = name
        self._subject = subject
        self._hemi = hemi

        if from_file:
            self._vertex_indexes, self._ras_coords = sfu.read_label(f'{SUBJECTS_DIR}/{subject.name}/label/{hemi}.{name}.label')
        else:
            self._vertex_indexes = vertex_indexes
            self._ras_coords = ras_coords
            
    @property
    def name(self):
        return self._name
    
    
    @property
    def subject(self):
        return self._subject
    
    @property
    def hemi(self):
        return self._subject.hemi
    
    @property
    def vertex_indexes(self):
        return self._vertex_indexes
    
    @property
    def ras_coords(self):
        return self._ras_coords
    
    @property
    def faces(self):
        label_faces = []
        for face in self._subject.faces:
            for vertex_index in face:
                if vertex_index in self._vertex_indexes:
                    label_faces.append(face)
                    
        return np.array(label_faces)
    
    @cached_property
    def mesh(self):
        return tm.Trimesh(vertices=self.subject.ras_coords, faces=self.faces)
    
    @cached_property
    def curv(self):
        return self._subject.curv
    
    def gyrus(self, curv_threshold=0):
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

    def sulcus(self, curv_threshold=0):
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
        


def create_label(name, subject, hemi="lh"):
    label = Label(name, subject, hemi)
    return label