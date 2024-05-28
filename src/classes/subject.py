
from functools import cached_property
from typing import List
import numpy as np
import nibabel as nb
from collections import defaultdict
from src.utilities import freesurfer_utils as fsu
from src.utilities import surface_utils
from src.utilities.plotting import initialize_scene, plot, plot_label, remove_label, show_scene

SUBJECTS_DIR = '/Users/benparker/Desktop/cnl/subjects'

class ScalpelSubject(object):
    
    def __init__(self, name, hemi="lh", surface_type="inflated", subjects_dir = SUBJECTS_DIR):
        self._name = name
        self._hemi = hemi
        self._surface_type = surface_type
        self._labels = defaultdict(list)
        self._subject_fs_path = f'{subjects_dir}/{name}/'
        self._scene = None
        self._mesh = {}
        
        surface = nb.freesurfer.read_geometry(f'{subjects_dir}/{name}/surf/{hemi}.{surface_type}')
        self._curv = nb.freesurfer.read_morph_data(f'{subjects_dir}/{self._name}/surf/{self._hemi}.curv')
        self._ras_coords, self._faces = surface[0], surface[1]
        self._gyrus = fsu.get_gyrus(np.unique(self._faces), self._ras_coords, self._curv)
        self._sulcus = fsu.get_sulcus(np.unique(self._faces), self._ras_coords, self._curv)

    ############################
    # Properties
    ############################    
    @property
    def name(self):
        ## Subject ID
        return self._name
    
    @property
    def hemi(self):
        ## Hemisphere
        return self._hemi
    
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
    
    @property
    def subject_fs_path(self):
        ## Path to subject's freesurfer directory
        return self._subject_fs_path
    
    @cached_property
    def mesh(self):
        gyrus_gray = [250, 250, 250]
        sulcus_gray = [130, 130, 130]
        if self.surface_type == 'inflated':
            print('Building cortical mesh (this may take 1 - 2 minutes)')
            gyrus_mesh = surface_utils.make_mesh(self._ras_coords, self._faces, self._gyrus[0], face_colors=gyrus_gray)
            sulcus_mesh = surface_utils.make_mesh(self._ras_coords, self._faces, self._sulcus[0], face_colors=sulcus_gray)
            self._mesh['gyrus'] = gyrus_mesh
            self._mesh['sulcus'] = sulcus_mesh
        else:
            self._mesh['cortex'] = surface_utils.make_mesh(self._ras_coords, self._faces, self.vertex_indexes, face_colors=gyrus_gray)
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
    
    def load_label(self, label_name, label_idxs=None, label_RAS=None):
        """
        Load a label into the subject instance. Either loads from file or from input parameters.

        Parameters:
        - label_name (str): Name of the label.
        - label_idxs (np.array, optional): Vertex indices of the label. Defaults to None.
        - label_RAS (np.array, optional): RAS coordinates of the label. Defaults to None.

        Returns:
        - None 
        """
        if label_idxs is not None:
            self._labels[label_name].append(label_idxs)
            self._labels[label_name].append(label_RAS)
        else:
            label_idxs, label_RAS = fsu.read_label(f'{self.subject_fs_path}/label/{self.hemi}.{label_name}.label')
            self._labels[label_name].append(label_idxs)
            self._labels[label_name].append(label_RAS)

    ############################
    # Visualization Methods
    ############################

    def plot(self, view='lateral', labels: List[str] = None):
        if self._mesh == {}:
            self.mesh
        if self._scene is None:
            self._scene = initialize_scene(self._mesh, view, self._hemi, self._surface_type)
        return plot(self._scene, view, labels, self.plot_label)

    def plot_label(self, label_name: str, view='lateral', label_ind=None, face_colors=None):
        return plot_label(self._scene, self._ras_coords, self._faces, self._labels, label_name, view, self._hemi, face_colors, label_ind)

    def remove_label(self, label_name: str):
        return remove_label(self._scene, label_name)

    def show(self):
        return show_scene(self._scene)

def create_subject(name, hemi="lh", surface_type="inflated"):
    subject = ScalpelSubject(name, hemi, surface_type)
    return subject
