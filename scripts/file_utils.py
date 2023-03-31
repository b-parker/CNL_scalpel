from pathlib import Path
import os
import glob
import subprocess as sp
import shlex

def freesurfer_label2annot(subject_path: str, label_list: list, hemi: str, ctab_path: str, annot_name: str, outdir: str = None):
    '''
    Runs freesurfer label2annot 
    INPUT:
    subject_path: str = filepath to subject's directory
    label_list: str = list of strings containing all desired labels
    hemi: str = hemisphere
    ctab_path: str = filepath to color table
    annot_name: str = desired label name to save annot
    outdir: str = directory to write the annot, DEFAULT /annots file within subjects /label 

    OUTPUT:
    annot of the location <outdir>/<hemi>.<annot_name>.annot
    '''
    ## Determine all paths exist
    assert Path(subject_path).exists(), f"Subject path does not exist: {subject_path}"
    assert Path(ctab_path).exists(), f"Color table does not exist: {ctab_path}"

    subject_path = Path(subject_path)
    ctab_path = Path(ctab_path)

    if not isinstance(outdir):
        outdir = Path(outdir)
    else:
        outdir = subject_path / 'annots'

    if not outdir.exists():
        outdir.mkdir()
    
    assert outdir.exists(), f"Outdir path does not exist: {outdir}"

    ## Sort labels into strings 

    label_for_cmd = []

    for label in label_list:
        label_for_cmd.append('--l')
        label_filename = hemi + label + '.label'
        label_for_cmd.append(label_filename)

    subject_id = subject_path.name
    all_labels = ' '.join(label_for_cmd)

    ## Generate and run command 

    cmd = f"mris_label2annot \
        --s {subject_id} \
        --ctab {ctab_path}\
        --a {annot_name} \
        --h {hemi} \
        {all_labels}"
    
    print(f'Calling: {cmd}')

    sp.check_call(shlex.splt(cmd))



def get_subjects_list(subjects_list: str, subjects_dir: str = os.environ('SUBJECTS_DIR')) -> list:
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
    



def sort_subjects_and_sulci(subject_filepaths: list, sulci_list: list, hemi: str) -> dict:
    '''
    Sorts subject hemispheres into groups based on which sulci are present in each hemisphere
    '''
    ## TODO Output a dictionary, key = colortable name: str = <hemi_sulci>, value: list = subject_id

    ### for subjects, check which paths exist and which dont, add to dictionary key based on presences
    for sub_path in subject_filepaths:
        subject_path = Path(sub_path)
        subject_id = subject_path.name
        assert subject_path.exists(), f"{subject_id} does not exist at {subject_path}"
        
        subject_label_paths = get_sulci_filepaths(sub_path, sulci_list, hemi)
        
        for i, label in enumerate(sulci_list):



def get_sulci_filepaths(subject_filepath: str, sulci_list: list, hemi: str) -> list:
    '''
    Takes a subject path, list of sulci, and hemisphere and returns list of sulci label paths
    '''      
    subject_filepath = Path(subject_filepath)
    assert subject_filepath.exists(), f"The subject file path does not exist: {subject_filepath}"

    label_paths = [subject_filepath / 'label' / f'{hemi}.{label}.label' for label in sulci_list]

    return label_paths










def create_freesurfer_ctab(labels: str, outdir: str, pallete: list = [] ):
    '''
    Creates a color table file for label2annot 
    '''




def annot_list_to_JSON():
    '''
    Takes a list of subjects for a project and their respective sulcal presence
    and saves them to JSON file
    '''



if __name__ == "__main__":
    main()
