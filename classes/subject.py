from functools import cached_property
import numpy as np
import nibabel as nb
import trimesh as tm
from collections import defaultdict
from src import freesurfer_utils as fsu
from src import surface_funcs 


# TODO: move global variable setup somewhere else!
SUBJECTS_DIR = '/Users/benparker/Desktop/cnl/subjects'

class Subject(object):
    
    def __init__(self, name, hemi="lh", surface_type="inflated"):
        self._name = name
        self._hemi = hemi
        self._surface_type = surface_type
        self._labels = defaultdict(list)
        self._subject_fs_path = f'{SUBJECTS_DIR}/{name}/'
        
        surface = nb.freesurfer.read_geometry(f'{SUBJECTS_DIR}/{name}/surf/{hemi}.{surface_type}')
        self._curv = nb.freesurfer.read_morph_data(f'{SUBJECTS_DIR}/{self._name}/surf/{self._hemi}.curv')
        self._ras_coords, self._faces = surface[0], surface[1]
        self._gyrus = fsu.get_gyrus(np.unique(self._faces), self._ras_coords, self._curv)
        self._sulcus = fsu.get_gyrus(np.unique(self._faces), self._ras_coords, self._curv)
        
    @property
    def name(self):
        return self._name
    
    @property
    def hemi(self):
        return self._hemi
    
    @property
    def surface_type(self):
        return self._surface_type
    
    @property
    def ras_coords(self):
        return self._ras_coords
    
    @property
    def faces(self):
        return self._faces
    
    @property
    def vertex_indexes(self):
        return np.unique(self.faces)
    
    @property
    def subject_fs_path(self):
        return self._subject_fs_path
    
    @cached_property
    def mesh(self):
        return tm.Trimesh(vertices=self._ras_coords, faces=self._faces)
    
    @cached_property
    def curv(self):
        return self._curv
    
    @cached_property
    def gyrus(self):
        return self._gyrus
    
    @cached_property
    def sulcus(self):
        return self._sulcus
    
    @property
    def labels(self):
        return self._labels
    
    def load_label(self, label_name, hemi, label_idxs=None, label_RAS=None):
        if label_idxs is not None:
            self._labels[label_name].append(label_idxs)
            self._labels[label_name].append(label_RAS)
        else:
            label_idxs, label_RAS = fsu.read_label(f'{self.subject_fs_path}/label/{hemi}.{label_name}.label')
            self._labels[label_name].append(label_idxs)
            self._labels[label_name].append(label_RAS)
    

def create_subject(name, hemi="lh", surface_type="inflated"):
    subject = Subject(name, hemi, surface_type)
    return subject