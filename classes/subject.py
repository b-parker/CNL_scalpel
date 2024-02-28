from functools import cached_property

import nibabel as nb
import trimesh as tm


# TODO: move global variable setup somewhere else!
SUBJECTS_DIR = '/Users/megananctil/WeinerLab/subjects'

class Subject(object):
    
    def __init__(self, name, hemi="lh", surface_type="inflated"):
        self._name = name
        self._hemi = hemi
        self._surface_type = surface_type
        
        surface = nb.freesurfer.read_geometry(f'{SUBJECTS_DIR}/{name}/surf/{hemi}.{surface_type}')
        self._ras_coords, self._faces = surface[0], surface[1]
        
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
    
    @cached_property
    def mesh(self):
        return tm.Trimesh(vertices=self._ras_coords, faces=self._faces)
    
    @cached_property
    def curv(self):
        return nb.freesurfer.read_morph_data(f'{SUBJECTS_DIR}/{self._name}/surf/{self._hemi}.curv')
    

def create_subject(name, hemi="lh", surface_type="inflated"):
    subject = Subject(name, hemi, surface_type)
    return subject