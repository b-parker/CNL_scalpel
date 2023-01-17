# Utilities
from pathlib import Path
import os
import subprocess as sp

# Data
import pandas as pd
import numpy as np

# Brain
import nibabel as nb
from nibabel.freesurfer.io import read_annot, read_label, read_morph_data
import cortex
#import src.mesh_laplace_sulci

## Load subject
os.environ['SUBJECTS_DIR'] = '/Users/benparker/Desktop/cnl/subjects'
os.environ['FREESURFER_HOME'] = '/Users/benparker/freesurfer'


project_dir = Path('/Users/benparker/Desktop/cnl/CNL_scalpel/results/')
subjects_dir = Path('/Users/benparker/Desktop/cnl/subjects/')
subjects_dir_str = '/Users/benparker/Desktop/cnl/subjects/'

## read_annot returns [0] labels at each vertex, -1 for no id [1]: ctab [2]: label names 
annot_verts, annot_ctab, annot_names = read_annot(subjects_dir / '100307/label/rh.aparc.a2009s.annot')


# import test subject to pycortex db
cortex.freesurfer.import_subj('100307', freesurfer_subject_dir=subjects_dir_str, whitematter_surf='white')


