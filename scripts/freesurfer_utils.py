from pathlib import Path
import os
import subprocess as sp
import shlex
import datetime
from numpy.random import randint
import json


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
    

    cmd = f"mris_label2annot \
        --s {subject_id} \
        --ctab {ctab_path}\
        --a {annot_name} \
        --h {hemi} \
        {all_labels}"
    
    print(f'Calling: {cmd}')

    sp.Popen(shlex.split(cmd))



def get_subjects_list(subjects_list: str, subjects_dir: str) -> list:
    '''
    Turns txt subject list into list of filepaths
    
    INPUTS:
    subjects_list: str = filepath to .txt subject list
    subjects_dir: str = filepath to subjects directory

    OUTPUTS:
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
    subject_filepath: list = output of get_subjects_list, a list of all full paths to subjects

    sulci_list : list = all possible sulci

    OUTPUT:
    subject_sulci_dict: dict = ={subject_id : [[lh_sulci_present, rh_sulci_present]]}
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
    
    INPUTS:
    ctab_name: str = desired name of color table
    label_list: list = list of strings containing all desired labels
    outdir: str = desired output directory
    pallete: list = custom colors - dict labels and rgb colors as strings, with rgb values separated by tab - i.e. ['MFS' : 'int<tab>int<tab>int', ...]
    '''
    
    outdir_path = Path(outdir)
    assert outdir_path.exists(), f"{outdir.resolve()} does not exist"

    ctab_path = ''.join([outdir, ctab_name, '.ctab'])
    date = datetime.datetime.now()

    if isinstance(palette, type(None)):
        palette = [f"{label} : {randint(low=1, high=248)} {randint(low=1, high=248)} {randint(low=1, high=248)}"  for label in label_list]
    else:
        assert len(palette) == len(label_list), f"Palette length does not match label list length"

    with open(ctab_path, 'w') as file:
        file.write(f'#$Id: {ctab_path}, v 1.38.2.1 {date.strftime("%y/%m/%d")} {date.hour}:{date.minute}:{date.second} CNL Exp $ \n')
        file.write(f"No. Label Name:                R   G   B   A\n")
        file.write(f"0  Unknown         0   0   0   0\n")
        for i, label_name in enumerate(label_list):
            file.write(f"{i + 1}    {label_name}                {palette[label_name]}  0\n")

    

def create_ctabs_from_dict(project_colortable_dir: str, json_file: str):
    ''' 
    Takes a dictionary of subjects and present sulci,
    creates a colortable file for each unique combination of sulci
    '''
    print(json_file)
    with open(json_file) as file:
        sulci_dict = json.load(file)


    all_sulci = list(sulci_dict.values())
    unique_sulci_lists = [list(sulc_list) for sulc_list in set(tuple(sulc_list) for sulc_list in all_sulci)]

    for sulci_list in unique_sulci_lists:
        ctab_name = '_'.join(sulci_list)
        print(f"Creating color table for {ctab_name}")
        create_freesurfer_ctab(ctab_name=ctab_name, label_list=sulci_list,
                               outdir=project_colortable_dir)
        
        

def dict_to_JSON(dictionary: dict, outdir: str, project_name: str):
    '''
    Takes a list of subjects for a project and their respective sulcal presence
    and saves them to JSON file

    INPUT:
    subject_sulci_dict: dict = dictionary of {hemi_subject_id, [sulci_list]} created by sort_subjects_and_sulci()
    outdir: str = write directory for json of colortables
            NOTE: should be written to project directory for colortables
    project_name: str = the name of the project to be the name of the .json i.e. voorhies_natcom_2021.json
    '''
    assert os.path.exists(outdir), f"{outdir} does not exist"
    
    save_file = os.path.join(outdir, f"{project_name}.json")

    with open(save_file, 'w') as file:
        json.dump(dictionary, file, indent=4)


def rename_labels(subjects_dir: str, subjects_list: str, sulci_dict: dict, by_copy: bool = True):
    '''
    Renames labels in a given hemisphere for all subjects in a given subjects list
    INPUT:
    subjects_dir: str = filepath to subjects directory
    subjects_list: str = filepath to subjects list
    sulci_list: dict = dict of sulci,{old_name: new_name}
    by_copy: bool = if True, copies files by cp (keeps original file) ; if False, renames files by mv (deletes original file)
    
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
                    








