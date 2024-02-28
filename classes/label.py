from functools import cached_property

import numpy as np
import src.freesurfer_utils as sfu
import trimesh as tm


# TODO: move global variable setup somewhere else!
SUBJECTS_DIR = '/Users/megananctil/WeinerLab/subjects'

class Label(object):
    
    def __init__(self, name, subject, hemi="lh"):
        
        # (MA) Not sure if we even want to feed in hemi for label creation if the subject
        # fed in will just restrict it anyway
        if subject.hemi != hemi:
            raise Exception(f'The loaded subject hemisphere is {subject.hemi}, please specify the correct hemisphere `lh` or `rh` for this label')
        self._name = name
        self._subject = subject
        self._hemi = hemi
        self._vertex_indexes, self._ras_coords = sfu.read_label(f'{SUBJECTS_DIR}/{subject.name}/label/{hemi}.{name}.label')
            
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


def create_label(name, subject, hemi="lh"):
    label = Label(name, subject, hemi)
    return label