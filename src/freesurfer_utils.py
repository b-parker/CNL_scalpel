from pathlib import Path
import os
import subprocess as sp
import shlex

import datetime
from numpy.random import randint
import json
import numpy as np
import tarfile
from nibabel.freesurfer.io import read_geometry
import pandas as pd
 
def freesurfer_label2annot(subjects_dir: str, subject_path: str, label_list: list, hemi: str, ctab_path: str, annot_name: str):
    '''
    Runs freesurfer label2annot 

    INPUT:
    subjects_dir: str = freesurfer subjects directory, os.environ['SUBEJCTS_DIR] called below
    subject_path: str = filepath to subject's directory
    label_list: str = list of strings containing all desired labels
    hemi: str = hemisphere
    ctab_path: str = filepath to color table
    annot_name: str = desired label name to save annot
    

    OUTPUT:
    annot of the location <outdir>/<hemi>.<annot_name>.annot
    '''
    ## Determine all paths exist
    assert Path(subject_path).exists(), f"Subject path does not exist: {subject_path}"
    assert Path(ctab_path).exists(), f"Color table does not exist: {ctab_path}"
    assert Path(subjects_dir).exists(), f"SUBJECTS_DIR does not exist: {subjects_dir}"

    os.environ['SUBJECTS_DIR'] = subjects_dir
    ctab_path = Path(ctab_path)

    ## Sort labels into strings 

    label_for_cmd = []

    for label in label_list:
        label_for_cmd.append('--l')
        label_filename = f"{hemi}.{label}.label"
        label_filepath = f"{subject_path}/label/{label_filename}"
        label_for_cmd.append(label_filepath)

    subject_id = os.path.basename(subject_path)
    all_labels = ' '.join(label_for_cmd)

    ## Generate and run command 
    my_env = {**os.environ, 'SUBJECTS_DIR' : f"{subjects_dir}"}

    cmd = f"mris_label2annot \
        --s {subject_id} \
        --ctab {ctab_path}\
        --a {annot_name} \
        --h {hemi} \
        {all_labels}"
    
    print(f'Calling: {cmd}')

    sp.Popen(shlex.split(cmd), env=my_env).wait()

def freesurfer_label2vol(subjects_dir : str, subject : str, hemi : str,  **kwargs):
    """ 
    Runs freesurfer's label2vol command : https://surfer.nmr.mgh.harvard.edu/fswiki/mri_label2vol

    Defaults to run with registration to the same subject. This is coded as the --identity flag registering to the identity mat

    INPUT:
    subjects_dir : str = filepath to freesurfer subjects dir
    subject : str = freesurfer subject ID
    hemi : str = hemisphere 
    label_name : str = filepath to label file (do not include .label)
    annot_name : str = filepath to annot file (do not include .annot)
    outfile : str = outfile name (do not include .nii.gz)

    OUTPUT:
    Creates a volume of the label as a binary mask
    
    """
    ## Determine if label files or annot files exist
    
    
    if 'label_name' in kwargs:
        if isinstance(kwargs['label_name'], str):
            label_file = f"{subjects_dir}/{subject}/label/{hemi}.{kwargs['label_name']}.label"

            label_for_cmd = ['--l', label_file] 
            all_labels = ' '.join(label_for_cmd)
            
        if isinstance(kwargs['label_name'], list): 
            label_files = [f"{subjects_dir}/{subject}/label/{hemi}.{label}.label" for label in kwargs['label_name']] 
            for label_file in label_files:
                assert Path(label_file).exists(), f"Label file does not exist: {label_file}"

            label_for_cmd = [] 

            for i, label in kwargs['label_name']:
                label_for_cmd.append('--l')
                label_for_cmd.append(label_files[i])
                all_labels = ' '.join(label_for_cmd)


    if 'annot_name' in kwargs:
        if isinstance(kwargs['annot_name'], str):
            annot_file = f"{subjects_dir}/{subject}/label/{hemi}.{kwargs['annot_name']}.annot"
            assert Path(annot_file).exists(), f"annot_file does not exist: {annot_file}" 

            label_for_cmd = ['--annot', annot_file] 
            all_labels = ' '.join(label_for_cmd)

        if isinstance(kwargs['annot_name'], list):
            annot_files = [f"{subjects_dir}/{subject}/label/{hemi}.{annot}.label" for annot in kwargs['annot_name']] 
            for annot_file in annot_files:
                assert Path(annot_file).exists(), f"Annot file does not exist: {annot_file}"

            label_for_cmd = [] 

            for i, annot in kwargs['annot_name']:
                label_for_cmd.append('--annot')
                label_for_cmd.append(annot_files[i])
                all_labels = ' '.join(label_for_cmd)

    outfile = f"{subjects_dir}/{subject}/mri/{hemi}.{kwargs['outfile_name']}.nii.gz"
    os.chdir(f"{subjects_dir}/{subject}")

    my_env = {**os.environ, 'SUBJECTS_DIR' : f"{subjects_dir}"}
    cmd = f"mri_label2vol \
            --temp ./mri/orig.mgz \
            --o {outfile} \
            --subject {subject}\
            --hemi {hemi} \
            --identity \
            {all_labels} "
    
    print(f'Calling: {cmd}')

    sp.Popen(shlex.split(cmd), env = my_env).wait()

    


def get_subjects_list(subjects_list: str, subjects_dir: str) -> list:
    '''
    Turns txt subject list into list of filepaths
    
    INPUT:
    subjects_list: str = filepath to .txt subject list
    subjects_dir: str = filepath to subjects directory

    OUTPUT:
    subjects_filepaths: list: list of subject filepaths as strings
    '''
    
    with open(subjects_list) as list_file:
        subject_names = [line.rstrip() for line in list_file]

    subject_filepaths = []
   
    for subject in subject_names:
        subject_filepath = os.path.join(subjects_dir, subject)
        
        assert os.path.exists(subject_filepath), f"{subject} does not exist within SUBJECTS_DIR {subjects_dir}"

        subject_filepaths.append(subject_filepath)
    
    return subject_filepaths
    


def sort_subjects_and_sulci(subject_filepaths: list, sulci_list: list) -> dict:
    '''
    Sorts subject hemispheres into groups based on which sulci are present in each hemisphere

    INPUT:
    subject_filepath : list - output of get_subjects_list, a list of all full paths to subjects

    sulci_list : list - all possible sulci

    OUTPUT:
    subject_sulci_dict : dict - {subject_id : [[lh_sulci_present, rh_sulci_present]]}
    '''
    
    subject_sulci_dict = {}

    ### for subjects, check which paths exist and which dont

    for sub_path in subject_filepaths:
        for hemi in ['lh', 'rh']:
            subject_path = Path(sub_path)
            subject_id = subject_path.name
            assert subject_path.exists(), f"{subject_id} does not exist at {subject_path}"
            
            subject_label_paths = get_sulci_filepaths(sub_path, sulci_list, hemi)
            existing_subject_labels_by_hemi = []

            for i, label in enumerate(sulci_list):
                if subject_label_paths[i].exists():
                    #print(f"{subject_id} has the {hemi} {label} label")
                    existing_subject_labels_by_hemi.append(label)
                else:
                    #print(f"{subject_id} does not have the {hemi} {label} label")
                    pass
            
    
    ##  add to dictionary key fo subject_id based on label existenc
            subject_sulci_dict[f"{hemi}_{subject_id}"] = existing_subject_labels_by_hemi

    return subject_sulci_dict



def get_sulci_filepaths(subject_filepath: str, sulci_list: list, hemi: str) -> list:
    '''
    Takes a subject path, list of sulci, and hemisphere and returns list of sulci label paths
    '''      
    subject_filepath = Path(subject_filepath)
    assert subject_filepath.exists(), f"The subject file path does not exist: {subject_filepath}"

    label_paths = [subject_filepath / 'label' / f'{hemi}.{label}.label' for label in sulci_list]

    return label_paths



def create_freesurfer_ctab(ctab_name: str, label_list: str, outdir: str, palette: dict = None ):
    '''
    Creates a color table file for label2annot 
    
    INPUT:
    ctab_name : str - desired name of color table
    label_list : list - list of strings containing all desired labels
    outdir : str - desired output directory
    pallete : list - custom colors - dict labels and rgb colors as strings, with rgb values separated by tab - i.e. ['MFS' : 'int<tab>int<tab>int', ...]
    '''
    
    outdir_path = Path(outdir)
    assert outdir_path.exists(), f"{outdir.resolve()} does not exist"

    ctab_path = f"{outdir}/{ctab_name}.ctab"
    date = datetime.datetime.now()

    if palette == None:        
        palette = {f"{label}" : f"{randint(low=1, high=248)} {randint(low=1, high=248)} {randint(low=1, high=248)}"  for label in label_list}
    else:
        assert len(palette) == len(label_list), f"Palette length does not match label list length"

    with open(ctab_path, 'w') as file:
        file.write(f'#$Id: {ctab_path}, v 1.38.2.1 {date.strftime("%y/%m/%d")} {date.hour}:{date.minute}:{date.second} CNL Exp $ \n')
        file.write(f"No. Label Name:                R   G   B   A\n")
        file.write(f"0  Unknown         0   0   0   0\n")
        for i, label_name in enumerate(label_list):
            file.write(f"{i + 1}    {label_name}                {palette[label_name]}  0\n")

    

def create_ctabs_from_dict(project_colortable_dir: str, sulci_list: list, json_file: str, project_name : str, palette: dict = None):
    ''' 
    Takes a dictionary of subjects and present sulci,
    creates a colortable file for each unique combination of sulci

    INPUT:
    project_colortable_dir : str - filepath to project colortable directory
    json_file : str - filepath to json file containing subject sulci dictionary
    sulci_list : list - list of all possible sulci
    palette : dict - custom colors - dict labels and rgb colors as strings, with rgb values separated by tab - i.e. ['MFS' : 'int<tab>int<tab>int', ...]
    project_name : str - unique identifier for project 
    '''
    print(json_file)
    with open(json_file) as file:
        sulci_dict = json.load(file)

    # get all sulci in dictionary
    all_sulci_in_dict = list(sulci_dict.values())
    
    # get unique combinations of sulci 
    unique_sulci_lists = [list(sulc_list) for sulc_list in set(tuple(sulc_list) for sulc_list in all_sulci_in_dict)]
    
    if palette == None:        
        palette = {f"{label}" : f"{randint(low=1, high=248)} {randint(low=1, high=248)} {randint(low=1, high=248)}"  for label in sulci_list}
    else:
        print(palette.keys())
        print(sulci_list)
        
        assert len(palette.keys()) == len(sulci_list), f"Palette length does not match label list length"

    # store unique comnbinations of sulci in dictionary, with key by indexed combination number
    ctab_file_dict = {}

    # this is done to avoid file length limitations when having all sulci in filename (linux=255 bytes)
    # match subject hemi entry to value in the ctab_file_dict

    for i, unique_sulci_list in enumerate(unique_sulci_lists):
         num_sulci = len(unique_sulci_list)
         ctab_name = f'{project_name}_ctab_{i}_{num_sulci}_sulci'
         ctab_file_dict[ctab_name] = unique_sulci_list
    
    dict_to_JSON(dictionary = ctab_file_dict, outdir = project_colortable_dir, project_name = f"{project_name}_ctab_files")
    
    for key in ctab_file_dict.keys():
        create_freesurfer_ctab(ctab_name=key, label_list=sulci_list,
                            outdir=project_colortable_dir, palette=palette)
        
        

def dict_to_JSON(dictionary: dict, outdir: str, project_name: str):
    '''
    Takes a dictionary and saves as a JSON

    INPUT:
    dictionary : dict - dictionary of {hemi_subject_id, [sulci_list]} created by sort_subjects_and_sulci()
    outdir : str - write directory for json of colortables
            NOTE: should be written to project directory for colortables
    project_name : str - the name of the project to be the name of the .json i.e. voorhies_natcom_2021.json
    '''
    print(outdir)
    assert os.path.exists(outdir), f"{outdir} does not exist"
    
    save_file = os.path.join(outdir, f"{project_name}.json")

    with open(save_file, 'w') as file:
        json.dump(dictionary, file, indent=4)


def rename_labels(subjects_dir: str, subjects_list: str, sulci_dict: dict, by_copy: bool = True):
    '''
    Renames labels in a given hemisphere for all subjects in a given subjects list

    INPUT:
    subjects_dir : str - filepath to subjects directory
    subjects_list : str - filepath to subjects list
    sulci_list : dict - dict of sulci,{old_name: new_name}
    by_copy : bool - if True, copies files by cp (keeps original file) ; if False, renames files by mv (deletes original file)
    
    '''
    assert os.path.exists(subjects_dir), f"{subjects_dir} does not exist"
    assert os.path.exists(subjects_list), f"{subjects_list} does not exist"
    
    subject_filepaths = get_subjects_list(subjects_list, subjects_dir)
    
    if by_copy == True:
        # Copies files by cp (keeps original file)
        for subject_path in subject_filepaths:
    
            assert os.path.exists(subject_path), f"The subject does not exist at {subject_path}"

            for hemi in ['lh', 'rh']:
                for sulcus in sulci_dict.items():
                    cmd = f"cp {subject_path}/label/{hemi}.{sulcus[0]}.label {subject_path}/label/{hemi}.{sulcus[1]}.label"
                    print(f"Executing: {cmd}")
                    run_cmd = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)

                    out, err = run_cmd.communicate()

                    if run_cmd.returncode == 0:
                        pass
                    else:
                        print(f"out: {out}")
                        print(f"err: {err}")
                        print(f'Be sure that {hemi}.{sulcus[0]}.label exists in {subject_path}')
                    

    else:
        # Renames files by mv (removes original file)
        for subject_path in subject_filepaths:
    
            assert os.path.exists(subject_path), f"The subject does not exist at {subject_path}"

            for hemi in ['lh', 'rh']:
                for sulcus in sulci_dict.items():
                    
                    cmd = f"mv {subject_path}/label/{hemi}.{sulcus[0]}.label {subject_path}/label/{hemi}.{sulcus[1]}.label"
                    print(f"Executing: {cmd}")
                    run_cmd = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)

                    out, err = run_cmd.communicate()

                    if run_cmd.returncode == 0:
                        pass
                    else:
                        print(f"out: {out}")
                        print(f"err: {err}")
                        print(f'Be sure that {hemi}.{sulcus[0]}.label exists in {subject_path}')


def create_tar_from_subject_list(project_dir: str, tarfile_name: str, subject_list: str, subjects_dir: str): 
    """
    Creates a compressed .tar.gz file from a list of subjects recursively. 
        NOTE: This will add ALL files located in freesurfer subject directory
    INPUT:
    project_dir : str - filepath to project directory where tar will be written
    tarfile_name : str - name for tar archive
    subject_list : str - filepath to .txt list of subjects
    subjects_dir : str - filepath freesurfer subjects directory


    """
    # Get subject list
    subject_list = get_subjects_list(subjects_dir=subjects_dir, subjects_list=subject_list)
    
    # check suffix and remove
    if tarfile_name[-7:] == '.tar.gz':
        tarfile_name = tarfile_name[:-7]
    
    assert os.path.exists(project_dir), "{project_dir} does not exist"

    
    # Check if tar exists
    try:
        with tarfile.open(f"{project_dir}/{tarfile_name}.tar.gz", mode='x:gz') as tar:
            print(f'Creating {tarfile_name} \n')
            for subject_dir in subject_list:
                tar.add(subject_dir, recursive=True)
            print('tarfile created.')
    except FileExistsError:
        # if tar exists, confirm user wants to add new subjects to tar
        print(f'\n {tarfile_name}.tar.gz already exists. \n')

        add_to_tar = input('Do you want to add the subjects to this existing tarfile? [y/n] ').lower()
        if add_to_tar == 'y' or add_to_tar == 'yes':
         print('\nAdding\n')
         
         with tarfile.open(f"{project_dir}{tarfile_name}.tar.gz", mode='w:gz') as tar:
            for subject_dir in subject_list:
                tar.add(subject_dir, recursive=True)
        else: 
            print(f'\nSubjects not added to {tarfile_name}.\n')
      

def write_label(label_name : str, label_faces : np.array, verts : np.array, hemi : str, subject : str, subjects_dir : str, surface_type : str = 'white'):
    """
    Write a freesurfer label file 

    INPUT:
    label_name : str - name of label for save file
    label_faces : np.array - array of faces for label
    verts : np.array - array of all vertices in subject hemi; if None, will read in subject hemisphere from /surf/ file
    hemi : str - hemisphere of label
    subject : str - subject ID
    subjects_dir : str - filepath to freesurfer subjects directory
    surface_type : str - surface type for label (default = 'white')

    OUTPUT:
    writes label file to subject label directory
    """
    
    assert os.path.exists(subjects_dir), f'{subjects_dir} does not exist'
    subject_dir = f"{subjects_dir}/{subject}"
    assert os.path.exists(subject_dir), f'{subject_dir} does not exist'


    label_ind = np.unique(label_faces)
    if verts is None:
        hemi_surf = f"{subjects_dir}/{subject}/surf/{hemi}.{surface_type}"
        verts, faces = read_geometry(hemi_surf)

    label_verts = verts[label_ind]

    label_filename = f"{subjects_dir}/{subject}/label/{hemi}.{label_name}.label"

    with open(label_filename, 'w') as f:
        f.write(f"#!ascii label , from subject {subject} vox2ras=TkReg coords={surface_type} \n")
        f.write(f"{len(label_ind)} \n")
        for i, ind in enumerate(label_ind):
            f.write(f"{ind} {np.round(label_verts[i][0], decimals=3)} {np.round(label_verts[i][1], decimals=3)} {np.round(label_verts[i][2], decimals=3)} 0.0000000000 \n")



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
