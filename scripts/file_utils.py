from pathlib import Path
import os
import glob
import subprocess as sp
import shlex
import datetime
from numpy.random import randint
import json


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

    for sub_path in enumerate(subject_filepaths):
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
            #existing_subject_labels.append(existing_subject_labels_by_hemi)
    
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





def create_freesurfer_ctab(ctab_name: str, label_list: str, outdir: str, pallete: list = [] ):
    '''
    Creates a color table file for label2annot 
    TODO
    '''
    outdir_path = Path(outdir)
    assert outdir_path.exists(), f"{outdir.resolve()} does not exist"

    ctab_path = ''.join([outdir, ctab_name, '.ctab'])
    date = datetime.datetime.now()

    with open(ctab_path, 'w') as file:
        file.write(f'#$Id: {ctab_path}, v 1.38.2.1 {date.strftime("%y/%m/%d")} {date.hour}:{date.minute}:{date.second} CNL Exp $ \n')
        file.write(f"No. Label Name:                R   G   B   A\n")
        file.write(f"0  Unknown         0   0   0   0\n")
        for i, label_name in enumerate(label_list):
            file.write(f"{i + 1}    {label_name}                {randint(low=1, high=248)}  {randint(low=1, high=248)}  {randint(low=1, high=248)}  0\n")

    

def create_ctabs_from_dict(project_colortable_dir: str, json: str):
    ''' 
    Takes a dictionary of subjects and present sulci,
    creates a colortable file for each unique combination of sulci
    '''
    json_file = open(json)

    sulci_data = json.load(json_file)

    all_sulci = list(sulci_data.values())
    unique_sulci_lists = [list(sulc_list) for sulc_list in set(tuple(sulc_list) for sulc_list in all_sulci)]

    for sulci_list in unique_sulci_lists:
        ctab_name = ''.join(sulci_list)
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
    '''
    json_dict = json.dump(dictionary)
    outdir = Path(outdir)
    save_file = outdir / project_name
    assert outdir.exists(), f"{outdir} does not exist"

    with open(save_file, 'w') as file:
        file.write(json_dict)




def main():
    subjects_dir = '/Users/benparker/Desktop/cnl/subjects/'
    subject_list = get_subjects_list(subjects_list='/Users/benparker/Desktop/cnl/subjects/subjects_list.txt',
                                     subjects_dir=subjects_dir)
    
    sulci_list = ['POS', '2', '3', 'MCGS']

    sorted_sulci_dict = sort_subjects_and_sulci(subject_list, sulci_list=sulci_list)

    create_freesurfer_ctab('test_annot', sulci_list, subjects_dir)

    print(sorted_sulci_dict)


if __name__ == "__main__":
    main()
