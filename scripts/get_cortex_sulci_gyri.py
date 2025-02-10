from src.freesurfer_utils import get_gyrus, get_sulcus, read_label, new_write_label
from src import freesurfer_utils
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import nibabel as nib
from dataclasses import dataclass



def get_subjects_list(subjects_list: str, subjects_dir: str) -> list:
    '''
    Turns txt subject list into list of filepaths
    
    INPUT:
    subjects_list: str = filepath to .txt subject list
    subjects_dir: str = filepath to subjects directory

    OUTPUT:
    subjects_filepaths: list: list of subject filepaths as strings
    '''
    
    with open(subjects_list, 'r', encoding="utf-8") as list_file:
        subject_names = [line.rstrip() for line in list_file]

    subject_filepaths = []
   
    for subject in subject_names:
        subject_filepath = os.path.join(subjects_dir, subject)
        
        assert os.path.exists(subject_filepath), f"{subject} does not exist within SUBJECTS_DIR {subjects_dir}"

        subject_filepaths.append(subject_filepath)
    
    return subject_filepaths



def new_write_label(label_indexes: np.array, label_RAS: np.array, label_name: str, hemi: str, subject_dir: str or Path, overwrite: bool = False):
    """
    Write freesurfer label file from label indexes and RAS coordinates

    INPUT:
    _____
    label_indexes: np.array - numpy array of label indexes from src.read_label()
    label_RAS: np.array - numpy array of label RAS vertices from src.read_label()
    label_name: str - name of label
    hemi: str - hemisphere of label
    subject_dir: str or Path - path to subject directory
    
    """
    
    if isinstance(subject_dir, str):
        subject_dir = Path(subject_dir)
    

    label_filename = subject_dir / 'label' / f'{hemi}.{label_name}.label'
    
    if overwrite == False:
        assert not label_filename.exists(), f"{hemi}.{label_name} already exists for subject at {subject_dir.absolute()}"

    subject_id = subject_dir.name
    label_length = label_indexes.shape[0]

    print(f'Writing label {label_filename.name} for {subject_id}')
    
    with open(label_filename.absolute(), 'w') as label_file:
        label_file.writelines(f'#!ascii label  , from subject {subject_id} vox2ras=TkReg coords=white\n')
        label_file.writelines(f'{label_length}\n')
        for i in range(label_length):
            label_line = f"{label_indexes[i]} {label_RAS[i][0]} {label_RAS[i][1]} {label_RAS[i][2]} 0.0000000000 \n"
            label_file.write(label_line)

def get_sulcus(label_index: np.array, label_RAS: np.array, curv: np.array, curv_threshold: int = 0):
    """ 
    Returns all label indices and RAS coordinates for sulcus within freesurfer label

    INPUT:
    _____
    label_index: np.array - numpy array of label indexes from src.read_label()
    label_RAS: np.array - numpy array of label RAS vertices from src.read_label()
    curv: np.array - numpy array of curvature values from nb.freesurfer.read_morph_data()
    curv_threshold: int - value for thresholding curvature value

    OUTPUT:
    sulcus_index: np.array - numpy array of sulcus indexes from src.read_label()
    sulcus_RAS: np.array - numpy array of sulcus RAS vertices from src.read_label()

    """
        
    sulcus_index = []
    sulcus_RAS = []

    for point, RAS in zip(label_index, label_RAS):
        if not isinstance(point, int):
            point = int(point)

        if curv[point] > curv_threshold:
            sulcus_index.append(point)
            sulcus_RAS.append(RAS)
        else:
            continue
    return np.array(sulcus_index), np.array(sulcus_RAS)


def get_gyrus(label_index: np.array, label_RAS: np.array, curv: np.array, curv_threshold: int = 0):
    """ 
    Returns all label indices and RAS coordinates for gyrus within freesurfer label

    INPUT:
    _____
    label_index: np.array - numpy array of label indexes from src.read_label()
    label_RAS: np.array - numpy array of label RAS vertices from src.read_label()
    curv: np.array - numpy array of curvature values from nb.freesurfer.read_morph_data()
    curv_threshold: int - value for thresholding curvature value

    OUTPUT:
    gyrus_index: np.array - numpy array of gyrus indexes from src.read_label()
    gyrus_RAS: np.array - numpy array of gyrus RAS vertices from src.read_label()

    """
        
    gyrus_index = []
    gyrus_RAS = []

    for point, RAS in zip(label_index, label_RAS):
        if not isinstance(point, int):
            point = int(point)

        if curv[point] < curv_threshold:
            gyrus_index.append(point)
            gyrus_RAS.append(RAS)
        else:
            continue
    return gyrus_index, gyrus_RAS




    



def read_label(label_name):
    """
    Reads a freesurfer-style .label file (5 columns)
    
    Parameters
    ----------
    label_name: str 
    
    Returns 
    -------
    vertices: index of the vertex in the label np.array [n_vertices] 
    RAS_coords: columns are the X,Y,Z RAS coords associated with vertex number in the label, np.array [n_vertices, 3] 
    
    """
    
    # read label file, excluding first two lines of descriptor 
    df_label = pd.read_csv(label_name,skiprows=[0,1],header=None,names=['vertex','x_ras','y_ras','z_ras','stat'],delimiter='\s+')
    
    vertices = np.array(df_label.vertex) 
    RAS_coords = np.empty(shape = (vertices.shape[0], 3))
    RAS_coords[:,0] = df_label.x_ras
    RAS_coords[:,1] = df_label.y_ras
    RAS_coords[:,2] = df_label.z_ras
    
    return vertices, RAS_coords




def main():
    subjects_dir  = '/Users/benparker/Desktop/cnl/neurocluster/weiner/DevProso/subjects'
    subjects_list = subjects_dir + '/all_subjects_list.txt'
    subject_paths = freesurfer_utils.get_subjects_list(subjects_list = subjects_list, subjects_dir = subjects_dir)
    subjects = [Path(subject_path).name for subject_path in subject_paths]



    subjects_dir = '/Users/benparker/Desktop/cnl/neurocluster/weiner/DevProso/subjects'
    for subject_path in subject_paths:
        for hemi in ['lh', 'rh']:
            cortex_label = subject_path +f'/label/{hemi}.cortex.label'
            lh_cortex_ind, lh_cortex_vert = read_label(cortex_label)
            
            curvature_path = subject_path + f'/surf/{hemi}.curv'
            subject_curvature = nib.freesurfer.read_morph_data(curvature_path)
            gyri_label_ind, gyri_label_vert  = get_gyrus(lh_cortex_ind, lh_cortex_vert, subject_curvature)
            sulci_label_ind, sulci_label_vert  = get_sulcus(lh_cortex_ind, lh_cortex_vert, subject_curvature)

            new_write_label(np.array(gyri_label_ind), np.array(gyri_label_vert), 'cortex_gyri', f'{hemi}', subject_path)
            new_write_label(np.array(sulci_label_ind), np.array(sulci_label_vert), 'cortex_sulci', f'{hemi}', subject_path)


if __name__ == "__main__":
    main()