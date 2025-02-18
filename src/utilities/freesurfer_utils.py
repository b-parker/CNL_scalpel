""" System imports """
from pathlib import Path
import os
import subprocess as sp
import shlex
import tarfile
import json
import datetime
from numpy.random import randint
import numpy as np
import pandas as pd

from typing import Union, Optional, List, Tuple

try:
    from typing import NoneType
except ImportError:
    NoneType = type(None)
    


def freesurfer_label2label(source_subjects_dir:str, source_subject: str, 
                           target_subject_dir: str, target_subject: str, 
                           source_label: str, 
                           target_label_dir: str, target_label:str, 
                            hemi: str, 
                            regmethod: str = 'surface', 
                            unique_source = False,
                            source_label_dir: str = None):
    '''
    Runs freesurfer label2label

    Args:
    subjects_dir: str = freesurfer subjects directory, os.environ['SUBJECTS_DIR'] called below
    source_subject: str = source subject ID
    target_subject: str = target subject ID
    target_subject_dir : str = target subject directory
    source_label: str = source label name
    target_label: str = target label name (Just provide label name, final format will be <src_sub>.<hemi>.<label>.label)
    hemi: str = hemisphere
    regmethod: str = registration method
    unique_source: bool = use a unique source label location

    Returns:
    annot of the location <outdir>/<hemi>.<annot_name>.annot

    '''
   # Construct source label path based on whether custom directory provided
    if source_label_dir:
        source_label_path = Path(source_label_dir) / f"{source_subject}.{hemi}.{source_label}.label"
    else:
        source_label_path = Path(source_subjects_dir) / source_subject / "label" / f"{hemi}.{source_label}.label"
    
    # Construct target label path
    target_label_path = Path(target_label_dir) / f"{source_subject}.{hemi}.{target_label}.label"
    
    # Verify paths
    assert Path(source_subjects_dir).exists(), f"SUBJECTS_DIR does not exist: {source_subjects_dir}"
    assert Path(source_subjects_dir, source_subject).exists(), f"Source subject does not exist: {source_subject}"
    assert Path(target_subject_dir, target_subject).exists(), f"Target subject does not exist: {target_subject}"
    assert source_label_path.exists(), f"Source label does not exist: {source_label_path}"
    
    # Set environment and construct command
    os.environ['SUBJECTS_DIR'] = str(source_subjects_dir)
    cmd = f"""mri_label2label \
        --srcsubject fsaverage \
        --srclabel {source_label_path} \
        --trgsubject {target_subject} \
        --trglabel {target_label_path} \
        --hemi {hemi} \
        --regmethod {regmethod}
    """
    
    print(f'Executing: {cmd}')
    sp.run(shlex.split(cmd), check=True)  

    


def freesurfer_label2annot(subjects_dir: str, subject_path: str, 
                           label_list: list, hemi: str, ctab_path: str, annot_name: str):
    '''
    Runs freesurfer label2annot 

    Args:
    subjects_dir: str = freesurfer subjects directory, os.environ['SUBJECTS_DIR'] called below
    subject_path: str = filepath to subject's directory
    label_list: str = list of strings containing all desired labels
    hemi: str = hemisphere
    ctab_path: str = filepath to color table
    annot_name: str = desired label name to save annot
    

    Returns:
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
        --ldir {subject_path}/label\
        --a {annot_name} \
        --h {hemi} \
        {all_labels}"
    
    print(f'Calling: {cmd}')

    sp.Popen(shlex.split(cmd), env=my_env).wait()



## run annotation2label on set of labels
def freesurfer_annotation2label(subject_dir: str, subject_id: str, outdir: str = None, annot_name: str ='aparc.a2009s'):
    """
    Runs freesurfer annotation2label command : https://surfer.nmr.mgh.harvard.edu/fswiki/mri_annotation2label

    Args:
    subject_dir : str = filepath to freesurfer subjects dir
    subject_id : str = freesurfer subject ID
    label_names : list = list of label names to be converted to labels
    outdir : str = filepath to output directory
    annot_name : str = filepath to annot file (do not include .annot)

    Returns:
    Creates a label for each label name in the annotation
    """

     ## Check existence of subject and annotations
    subject_dir = Path(subject_dir)
    subject_path = subject_dir / subject_id 
    assert subject_path.exists(), f"Subject not found at {subject_path.absolute()}"
    annot_path = subject_path / "label" / f"lh.{annot_name}.annot" 
    assert annot_path.exists(), f"Annotation not found at {annot_path.absolute()}.annot"

    ## Set environment variables for the subject
    existing_env = os.environ.copy()
    existing_env["SUBJECTS_DIR"] = subject_dir.absolute()

    if outdir == None:
        outdir = subject_path / "/label"

    outdir = Path(outdir)

    ## Create command for annotation - append all labels in label_names
    for hemi in ['lh', 'rh']:
        command = ['mri_annotation2label',
                    f'--subject {subject_id}',
                    f'--hemi {hemi}',
                    f'--annotation {annot_name}',
                    f'--outdir {outdir}']
         
        print("COMMAND:", " ".join(command)) 
        ## 
        cmd_open = sp.Popen(' '.join(command), env = existing_env, stderr=sp.PIPE, stdout=sp.PIPE, shell=True)

        stdout, stderr = cmd_open.communicate()  

        if cmd_open.returncode == 0:
              print("Command succeeded with Returns:")
              print(stdout.decode())
        else:
              print("Command failed with error:")
              print(stderr.decode())

def freesurfer_mris_anatomical_stats(
    subject_name: str,
    hemisphere: str,
    subjects_dir: str,
    surface_name: Optional[str] = None,
    thickness_range: Optional[tuple[float, float]] = None,
    label_file: Optional[Union[str, Path]] = None,
    thickness_file: Optional[Union[str, Path]] = None,
    annotation_file: Optional[Union[str, Path]] = None,
    tabular_Returns: bool = False,
    table_file: Optional[Union[str, Path]] = None,
    log_file: Optional[Union[str, Path]] = None,
    smooth_iterations: Optional[int] = None,
    color_table_file: Optional[Union[str, Path]] = None,
    no_global: bool = False,
    th3: bool = False,
    freesurfer_home = os.environ.get('FREESURFER_HOME')
) -> sp.CompletedProcess:
    """
    Run mris_anatomical_stats command with specified parameters.
    
    Args:
        subject_name: Subject name
        hemisphere: Hemisphere ('lh' or 'rh')
        subjects_dir: Path to subjects directory
        surface_name: Optional surface name
        thickness_range: Tuple of (low_thresh, high_thresh) for thickness consideration
        label_file: Path to label file
        thickness_file: Path to thickness file
        annotation_file: Path to annotation file
        tabular_Returns: Whether to use tabular output format
        table_file: Path to output table file
        log_file: Path to log file
        smooth_iterations: Number of smoothing iterations
        color_table_file: Path to output color table file
        no_global: Whether to skip global brain stats
        th3: Whether to compute vertex-wise volume using tetrahedra

    Returns:
        CompletedProcess instance with return code and output
    
    Raises:
        sp.CalledProcessError: If the command returns non-zero exit status
    """

    if not freesurfer_home:
        freesurfer_home = os.environ.get('FREESURFER_HOME')
    
    if freesurfer_home is None:
        RaiseValueError("FREESURFER_HOME not set in environment or passed as argument")

    # Build command list
    cmd = ['mris_anatomical_stats']
    
    if not isinstance(subjects_dir, str):
        subjects_dir = str(subjects_dir)
        os.environ['SUBJECTS_DIR'] = subjects_dir
    
    env = os.environ.copy()
    env['SUBJECTS_DIR'] = subjects_dir
    env['FREESURFER_HOME'] = freesurfer_home
    # Clean any None values from the environment
    env = {k: str(v) for k, v in env.items() if v is not None}
    

    print(f"SUBJECTS_DIR: {os.environ['SUBJECTS_DIR']}")

    # Add optional flagged arguments
    if thickness_range:
        cmd.extend(['-i', str(thickness_range[0]), str(thickness_range[1])])
    
    if label_file:
        cmd.extend(['-l', str(label_file)])
    
    if thickness_file:
        cmd.extend(['-t', str(thickness_file)])
    
    if annotation_file:
        cmd.extend(['-a', str(annotation_file)])
    
    if tabular_Returns:
        cmd.append('-b')
    
    if table_file:
        cmd.extend(['-f', str(table_file)])
    
    if log_file:
        cmd.extend(['-log', str(log_file)])
    
    if smooth_iterations is not None:
        cmd.extend(['-nsmooth', str(smooth_iterations)])
    
    if color_table_file:
        cmd.extend(['-c', str(color_table_file)])
    
    if no_global:
        cmd.append('-noglobal')
    
    if th3:
        cmd.append('-th3')
    
    # Add required positional arguments
    cmd.extend([subject_name, hemisphere])
    
    # Add optional positional argument
    if surface_name:
        cmd.append(surface_name)
    
    try: 
        # Run command
        return sp.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            env=env
        )

    except sp.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Command: {e.cmd}")
        print(f"Error output: {e.stderr}")
        raise


def freesurfer_label2vol(subjects_dir : str, subject : str, hemi : str, outfile_name : str, outfile_subjects_dir = None,  **kwargs):
    """ 
    Runs freesurfer's label2vol command : https://surfer.nmr.mgh.harvard.edu/fswiki/mri_label2vol

    Defaults to run with registration to the same subject. This is coded as the --identity flag registering to the identity mat

    Args:
    subjects_dir : str = filepath to freesurfer subjects dir
    subject : str = freesurfer subject ID
    hemi : str = hemisphere 
    outfile_name : str = outfile name (do not include .nii.gz)
    label_name : str = filepath to label file (do not include .label)
    annot_name : str = filepath to annot file (do not include .annot)
    

    Returns:
    Creates a volume of the label as a binary mask
    
    """
    ## Determine if label files or annot files exist
    
    
    if 'label_name' in kwargs:
        if isinstance(kwargs['label_name'], str):
            label_file = f"{subjects_dir}/{subject}/label/{hemi}.{kwargs['label_name']}.label"

            label_for_cmd = ['--label', label_file] 
            all_labels = ' '.join(label_for_cmd)
            
        if isinstance(kwargs['label_name'], list): 
            label_files = [f"{subjects_dir}/{subject}/label/{hemi}.{label}.label" for label in kwargs['label_name']] 
            for label_file in label_files:
                assert Path(label_file).exists(), f"Label file does not exist: {label_file}"

            label_for_cmd = [] 

            for i, label in kwargs['label_name']:
                label_for_cmd.append('--label')
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

    if isinstance(outfile_subjects_dir, NoneType):
        outfile = f"{subjects_dir}/{subject}/mri/{hemi}.{outfile_name}.nii.gz"
        os.chdir(f"{subjects_dir}/{subject}")
    else:
        outfile = f"{outfile_subjects_dir}/{subject}/mri/{hemi}.{outfile_name}.nii.gz"

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
    Takes a txt subject list and returns a list of subject directory filepaths
    
    Args:
    subjects_list: str = filepath to .txt subject list
    subjects_dir: str = filepath to subjects directory

    Returns:
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
    


def sort_subjects_and_sulci(subject_filepaths: list, sulci_list: list) -> dict:
    '''
    Sorts subject hemispheres into groups based on which sulci are present in each hemisphere

    Args:
    subject_filepath : list - output of get_subjects_list, a list of all full paths to subjects

    sulci_list : list - all possible sulci

    Returns:
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



def create_freesurfer_ctab(ctab_name: str, label_list: list, outdir: str, palette: dict = None ):
    '''
    Creates a color table file for label2annot 
    
    Args:
    ctab_name : str - desired name of color table
    label_list : list - list of strings containing all desired labels
    outdir : str - desired output directory
    pallete : list - custom colors - dict labels and rgb colors as strings, with rgb values separated by tab - i.e. ['MFS' : 'int<tab>int<tab>int', ...]
    '''
    
    outdir_path = Path(outdir)
    assert outdir_path.exists(), f"{outdir.resolve()} does not exist"

    ctab_path = f"{outdir}/{ctab_name}.ctab"
    date = datetime.datetime.now()

    if palette is None:        
        palette = {f"{label}" : f"{randint(low=1, high=248)} {randint(low=1, high=248)} {randint(low=1, high=248)}"  for label in label_list}
    else:
        pass

    with open(ctab_path, 'w', encoding="utf-8") as file:
        file.write(f'#$Id: {ctab_path}, v 1.38.2.1 {date.strftime("%y/%m/%d")} {date.hour}:{date.minute}:{date.second} CNL Exp $ \n')
        file.write("No. Label Name:                R   G   B   A\n")
        file.write("0  Unknown         0   0   0   0\n")
        for i, label_name in enumerate(label_list):
            file.write(f"{i + 1}    {label_name}                {palette[label_name]}  0\n")

    

def create_ctabs_from_dict(project_colortable_dir: str, sulci_list: list, json_file: str, project_name : str, palette: dict = None):
    ''' 
    Takes a dictionary of subjects and present sulci,
    creates a colortable file for each unique combination of sulci

    Args:
    project_colortable_dir : str - filepath to project colortable directory
    json_file : str - filepath to json file containing subject sulci dictionary
    sulci_list : list - list of all possible sulci
    palette : dict - custom colors - dict labels and rgb colors as strings, with rgb values separated by tab - i.e. ['MFS' : 'int<tab>int<tab>int', ...]
    project_name : str - unique identifier for project 

    Returns:
    ctab files for each unique combination of sulci
    '''
    print(json_file)
    with open(json_file, 'r', encoding="utf-8") as file:
        sulci_dict = json.load(file)

    # get all sulci in dictionary
    all_sulci_in_dict = list(sulci_dict.values())
    
    # get unique combinations of sulci 
    unique_sulci_lists = [list(sulc_list) for sulc_list in set(tuple(sulc_list) for sulc_list in all_sulci_in_dict)]
    
    if palette is None:        
        palette = {f"{label}" : f"{randint(low=1, high=248)} {randint(low=1, high=248)} {randint(low=1, high=248)}"  for label in sulci_list}
    else:
        print(palette.keys())
        print(sulci_list)
        
        assert len(palette.keys()) == len(sulci_list), "Palette length does not match label list length"

    # store unique comnbinations of sulci in dictionary, with key by indexed combination number
    ctab_file_dict = {}

    # this is done to avoid file length limitations when having all sulci in filename (linux=255 bytes)
    # match subject hemi entry to value in the ctab_file_dict

    for i, unique_sulci_list in enumerate(unique_sulci_lists):
        num_sulci = len(unique_sulci_list)
        ctab_name = f'{project_name}_ctab_{i}_{num_sulci}_sulci'
        ctab_file_dict[ctab_name] = unique_sulci_list
    
    dict_to_json(dictionary = ctab_file_dict, outdir = project_colortable_dir, project_name = f"{project_name}_ctab_files")
    
    # Get custom palette for each sulcus
    for key, value in ctab_file_dict.items():
        custom_palette = dict((val, palette[val]) for val in value)
        create_freesurfer_ctab(ctab_name=key, label_list=value,
                            outdir=project_colortable_dir, palette=custom_palette)

        
        

def dict_to_json(dictionary: dict, outdir: str, project_name: str):
    '''
    Takes a dictionary and saves as a JSON

    Args:
    dictionary : dict - dictionary of {hemi_subject_id, [sulci_list]} created by sort_subjects_and_sulci()
    outdir : str - write directory for json of colortables
            NOTE: should be written to project directory for colortables
    project_name : str - the name of the project to be the name of the .json i.e. voorhies_natcom_2021.json

    Returns:
    .json file of dictionary
    '''
    print(outdir)
    assert os.path.exists(outdir), f"{outdir} does not exist"
    
    save_file = os.path.join(outdir, f"{project_name}.json")

    with open(save_file, 'w', encoding="utf-8") as file:
        json.dump(dictionary, file, indent=4)


def rename_labels(subjects_dir: str, subjects_list: str, sulci_dict: dict, by_copy: bool = True):
    '''
    Renames labels in a given hemisphere for all subjects in a given subjects list

    Args:
    subjects_dir : str - filepath to subjects directory
    subjects_list : str - filepath to subjects list
    sulci_list : dict - dict of sulci,{old_name: new_name}
    by_copy : bool - if True, copies files by cp (keeps original file) ; if False, renames files by mv (deletes original file)

    Returns:
    Renamed label files
    
    '''
    assert os.path.exists(subjects_dir), f"{subjects_dir} does not exist"
    assert os.path.exists(subjects_list), f"{subjects_list} does not exist"
    
    subject_filepaths = get_subjects_list(subjects_list, subjects_dir)
    
    if by_copy is True:
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
    Args:
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
        if add_to_tar in ['y', 'yes']:
         print('\nAdding\n')
        
         with tarfile.open(f"{project_dir}{tarfile_name}.tar.gz", mode='w:gz') as tar:
            for subject_dir in subject_list:
                tar.add(subject_dir, recursive=True)
        else: 
            print(f'\nSubjects not added to {tarfile_name}.\n')


def create_tar_for_file_from_subject_list(project_dir: str, tarfile_name: str, subject_list: str, subjects_dir: str, filepath_from_subject_dir : str): 
    """
    Creates a compressed .tar.gz file from a list of subjects recursively. 
        NOTE: This will add ALL files located in freesurfer subject directory
    Args:
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
                
                tar.add(f"{subject_dir}/{filepath_from_subject_dir}")
            print('tarfile created.')
    except FileExistsError:
        # if tar exists, confirm user wants to add new subjects to tar
        print(f'\n {tarfile_name}.tar.gz already exists. \n')

        add_to_tar = input('Do you want to add the subjects to this existing tarfile? [y/n] ').lower()
        if add_to_tar in ['y', 'yes']:
         print('\nAdding\n')
        
         with tarfile.open(f"{project_dir}{tarfile_name}.tar.gz", mode='w:gz') as tar:
            for subject_dir in subject_list:
                tar.add(f"{subject_dir}/{filepath_from_subject_dir}")
        else: 
            print(f'\nSubjects not added to {tarfile_name}.\n')


def read_label(label_name: Union[str, Path], include_stat: bool = False) -> tuple:
    """
    Reads a freesurfer-style .label file (5 columns)
    
    Parameters
    ----------
    label_name: str or Path - name of label file to be read
    
    Returns 
    -------
    vertices: index of the vertex in the label np.array [n_vertices] 
    RAS_coords: columns are the X,Y,Z RAS coords associated with vertex number in the label, np.array [n_vertices, 3] 
    incude_stat: bool - if True, includes the statistic / value of the vertex in the label (fifth column in .label file)
    
    """
    
    # read label file, excluding first two lines of descriptor 
    df_label = pd.read_csv(label_name,skiprows=[0,1],header=None,names=['vertex','x_ras','y_ras','z_ras','stat'],delimiter='\s+')
    
    vertices = np.array(df_label.vertex) 
    RAS_coords = np.empty(shape = (vertices.shape[0], 3))
    RAS_coords[:,0] = df_label.x_ras
    RAS_coords[:,1] = df_label.y_ras
    RAS_coords[:,2] = df_label.z_ras

    if include_stat:
        stat = df_label.stat
        return vertices, RAS_coords, stat
    
    return vertices, RAS_coords

def write_label( label_name: Union[str, Path], label_indexes: np.array, label_RAS: np.array, hemi: str, subject_dir: Union[str, Path], surface_type: str, overwrite: bool = False, **kwargs):
    """
    Write freesurfer label file from label indexes and RAS coordinates

    Args:
    label_name: str - name of label to be written
    label_indexes: np.array - numpy array of label indexes
    label_RAS: np.array - numpy array of label RAS coordinates
    hemi: str - hemisphere
    subject_dir: str or Path - subject directory
    surface_type: str - surface type for label
    overwrite: bool - overwrite existing label file
    custom_label_name: str - custom label name for label file (optional) - shhould be complete string literal of label name in subject label file i.e. 'custom.label.name.label'
    custom_label_dir: str - custom label directory for label file (optional) - should be complete string literal of label directory in subject label file
    
    """
    
    if isinstance(subject_dir, str):
        subject_dir = Path(subject_dir)
    
    ## Check for custom label directory
    if 'custom_label_dir' in kwargs:
        if isinstance(kwargs['custom_label_dir'], str):
            label_dir = Path(kwargs['custom_label_dir'])
        else:
            label_dir = kwargs['custom_label_dir']
    else:
        label_dir = subject_dir / 'label'

    ## Check for custom label name
    if 'custom_label_name' in kwargs:
        label_name = kwargs['custom_label_name']
    else:
        label_name = f"{hemi}.{label_name}.label"
    
    ## Create full filename
    label_filename = label_dir / label_name
    
    if not overwrite:
        assert not label_filename.exists(), f"{hemi}.{label_name} already exists for subject at {subject_dir.absolute()}"

    subject_id = subject_dir.name
    label_length = label_indexes.shape[0]

    print(f'Writing label {label_filename.name} for {subject_id}')
    
    with open(label_filename.absolute(), 'w') as label_file:
        label_file.writelines(f'#!ascii label  , from subject {subject_id} vox2ras=TkReg coords={surface_type}\n')
        label_file.writelines(f'{label_length}\n')
        for i in range(label_length):
            label_line = f"{label_indexes[i]} {label_RAS[i][0]} {label_RAS[i][1]} {label_RAS[i][2]} 0.0000000000 \n"
            label_file.write(label_line)

def get_sulcus(label_index: np.array, label_RAS: np.array, curv: np.array, curv_threshold: int = 0):
    """ 
    Returns all label indices and RAS coordinates for sulcus within freesurfer label

    Args:
    _____
    label_index: np.array - numpy array of label indexes from src.read_label()
    label_RAS: np.array - numpy array of label RAS vertices from src.read_label()
    curv: np.array - numpy array of curvature values from nb.freesurfer.read_morph_data()
    curv_threshold: int - value for thresholding curvature value

    Returns:
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

    Args:
    _____
    label_index: np.array - numpy array of label indexes from src.read_label()
    label_RAS: np.array - numpy array of label RAS vertices from src.read_label()
    curv: np.array - numpy array of curvature values from nb.freesurfer.read_morph_data()
    curv_threshold: int - value for thresholding curvature value

    Returns:
    gyrus_index: np.array - numpy array of gyrus indexes from src.read_label()
    gyrus_RAS: np.array - numpy array of gyrus RAS vertices from src.read_label()

    """
        
    gyrus_index = []
    gyrus_RAS = []

    for point, RAS in zip(label_index, label_RAS):
        if curv[point] < curv_threshold:
            gyrus_index.append(point)
            gyrus_RAS.append(RAS)
        else:
            continue
    return np.array(gyrus_index), np.array(gyrus_RAS)


def mris_anatomical_stats2DataFrame_row(subject: str, label_name: str, hemi: str, data_dir: Union[str, Path]) -> pd.DataFrame:
    """ 
    Takes a subject list and the location of a stats.txt file outputted by mris_anatomical_stats ->> converts it to a dataframe

    Args:
    subject: str - subject ID
    label_name: str - name of the label to be included in the dataframe
    hemi: str - hemisphere to be included in the dataframe (must be 'lh', 'rh')
    data_dir: str or Path - directory where the stats.txt file is located

    Returns:
    pd.DataFrame


    """

    assert hemi in ['lh', 'rh'], "hemi must be 'lh' or 'rh'"

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    txt_path = data_dir / f"{hemi}.{label_name}.label.stats.txt"
    assert txt_path.exists(), f'the file {txt_path} does not exist.'

    all_stats_df = pd.DataFrame(columns=['sub', 'hemi', 'label', 'num_vertices', 'surface_area_mm^2', 'gray_matter_volume_mm^3', 'avg_cortical_thickness', 'avg_cortical_thickness_std', 'rectified_mean_curvature', 'rectified_gaussian_curvature', 'folding_index', 'intrinsic_curvature'])

    with open(txt_path, 'r') as fp:
        new_surf = fp.readlines()
    
    row_stats = new_surf[-1]
    row_stats = row_stats.split(' ')
    row_stats = [i for i in row_stats if i != '']
    label_name = row_stats[-1][:-1].split('.')[1]

    num_rows =  [row_stats[0], row_stats[1], row_stats[2], row_stats[3], row_stats[4], row_stats[5], row_stats[6], row_stats[7], row_stats[8]]
    num_rows = [float(i) for i in num_rows]
    all_stats_row = [subject, hemi, label_name, num_rows[0], num_rows[1], num_rows[2], num_rows[3], num_rows[4], num_rows[5], num_rows[6], num_rows[7], num_rows[8]]
    
    all_stats_df.loc[len(all_stats_df)] = all_stats_row

    return all_stats_df


def subject_label_stats2DataFrame(subjects_dir: Union[str, Path], subject_list: list, label_name: Union[str, list], hemi: Union[str, list], data_dir_from_subject_fs_dir:str = 'label', must_exist = True) -> pd.DataFrame:
    """ 
    Takes a subject list, label list, and the location of a stats.txt file outputted by mris_anatomical_stats ->> converts it to a dataframe

    Args:
    subjects_dir: str or Path - FreeSurfer subjects directory
    subject_list: list - list of subjects to be included in the dataframe
    label_name: str or list - name of the label to be included in the dataframe
    hemi: str or list - hemisphere to be included in the dataframe (must be 'lh', 'rh', or 'both')
    data_dir_from_subject_fs_dir: str - directory where the stats.txt file is located relative to the subject's FreeSurfer directory, default is 'label'

    Returns: 
    pd.DataFrame

    """

    assert hemi in ['lh', 'rh', 'both'], "hemi must be 'lh', 'rh', or 'both'"
    if hemi == 'both':
        hemi_list = ['lh', 'rh']
    else:
        hemi_list = [hemi]

    if isinstance(label_name, str):
        label_name = [label_name]
        
    if isinstance(subjects_dir, str):
        subjects_dir = Path(subjects_dir)

    all_stats_df = pd.DataFrame(columns=['sub', 'hemi', 'label', 'num_vertices', 'surface_area_mm^2', 'gray_matter_volume_mm^3', 'avg_cortical_thickness', 'avg_cortical_thickness_std', 'rectified_mean_curvature', 'rectified_gaussian_curvature', 'folding_index', 'intrinsic_curvature'])

    
    for sub in subject_list:
        for hemi in hemi_list:
            for label in label_name:
                data_dir = subjects_dir / sub / data_dir_from_subject_fs_dir 
                if must_exist:
                    assert data_dir.exists(), f"{data_dir} does not exist"
                else:
                    try:
                        new_row = mris_anatomical_stats2DataFrame_row(sub, label, hemi, data_dir)
                        print(new_row)
                        all_stats_df = pd.concat([all_stats_df, new_row], axis = 0)
                    except:
                        pass

    return all_stats_df


############################################################################################################
############################################################################################################
# Maximum Probability Map
############################################################################################################
############################################################################################################

## Project labels to fsaverage

    ## Create probabilty maps for each subject, with that subject held out
def create_prob_label(project_id: str, fsaverage_projected_label_dir: str, subject_list_path: str, prob_map_label_dir: str, label_name: str, left_out_subject: str,  hemi: str):
            """ 
            Creates probabilistic label files for a given label, with a subject held out

            Args:
            project_id: str - unique identifier for project (included in final name of probabilistic label)
            fsaverage_projected_label_dir: str - filepath to fsaverage projected labels (resulting from freesurfer_label2label)
            subject_list_path: str - filepath to subject list
            prob_map_label_dir: str - filepath to probabilistic label directory
            label_name: str - name of the label
            left_out_subject: str - subject to be held out
            hemi: str - hemisphere

            Returns:
            probabilistic label files for each subject, with the left out subject held out 
            """
            ## Load subjects and remove left out subject
            subjects = np.genfromtxt(subject_list_path) 

            subjects = [i for i in subjects if i != left_out_subject]

            ## Create empty array for vertices of projected labels
            label_vertices= np.empty(shape=0,dtype=int)
            label_RAS = np.empty(shape=(0,3),dtype=int)

            prob_label_name = f'{hemi}.{project_id}_PROB_{label_name}.label'
            prob_label_dir = prob_map_label_dir + f'/{left_out_subject}/'
            os.makedirs(prob_label_dir, exist_ok=True)

            ## Loop through subjects, load projected labels, and append vertices and RAS coords to arrays
            for sub in subjects:
                # Load projected label
                label_path = fsaverage_projected_label_dir + f'/projected_labels/{label_name}/{sub}.lh.{label_name}.label'
                vertices, RAS = read_label(label_path)
                
                # Append vertices from projected label to array
                label_vertices = np.append(label_vertices, vertices)
                label_RAS = np.append(label_RAS,RAS,axis=0)
                
                # Update unique vertices and counts
                unique_vertices, indices_vertices, counts_vertices=np.unique(label_vertices,return_index=True,return_counts=True)

                # index only the RAS coords for unique vertices
                unique_RAS = label_RAS[indices_vertices,:]

                # get probabilities at each vertex 
                prob_vertices = (counts_vertices)/len(subjects)
                

            # make probabilistic label array for label file
            prob_array = np.zeros(shape=(unique_vertices.shape[0],5),dtype=float)
            prob_array[:,0] = unique_vertices
            prob_array[:,1:4] = unique_RAS
            prob_array[:,-1] = prob_vertices

            # write_label(label_name = label_name, label_indices = unique_vertices, label_RAS = unique_RAS, hemi = hemi, 
            #             custom_label_dir = prob_map_label_dir, custom_label_name = prob_label_name) 
            #     # np.savetxt(prob_label_path, prob_array, fmt='%-2d  %2.3f  %2.3f  %2.3f %1.10f')

            # edit first two lines of label file to match Freesurfer
            f = open(prob_label_path, 'r')
            edit_f = f.read()
            f.close()
            f = open(prob_label_path, 'w')
            f.write('#!ascii label  , from subject fsaverage vox2ras=TkReg\n{}\n'.format(unique_vertices.shape[0]))
            f.write(edit_f)
            f.close()

## Create Maximum Probability Map
def create_MPM(subjects_dir: str, subjects_list: str, fsaverage_space_labels: str, project_id: str):
    """
    """

    prob_maps_vertices = {}
    prob_maps_RAS = {}
    prob_maps_stat = {}

    MPM_vertices = {}
    MPM_RAS = {}
    MPM_stat = {}

    print('Creating Left Hemisphere MPMs \n\n\n')

    # loop through labels, load prob map and make empty values in MPM dicts
    for i, middle_frontal_label in enumerate(middle_frontal_label_names):
        try:
            #load the prob mpa for the given label
            vertices, RAS, stat = read_label(fsaverage_space_labels + 'prob_maps/{}/lh.{}_PROB_{}.label'.format(left_out_sub, project_id, middle_frontal_label))
            prob_maps_vertices[middle_frontal_label] = vertices
            prob_maps_RAS[middle_frontal_label] = RAS
            prob_maps_stat[middle_frontal_label] = stat

            MPM_vertices[middle_frontal_label] = np.empty(0)
            MPM_RAS[middle_frontal_label] =  np.empty(shape=(0,3))
            MPM_stat[middle_frontal_label] = np.empty(0)
        except Exception:
            pass
            #load lh cortex vertices
    vertices_lh, RAS_lh = read_label('/home/weiner/data/fsaverage/label/lh.cortex.label')
        

    # loop through lh cortex vertices
    for vtx in vertices_lh:

        labels_with_vtx = np.empty(0)
        vertices_with_vtx = np.empty(0)
        RAS_with_vtx = np.empty(shape=(0,3))
        stat_with_vtx = np.empty(0)

        for label_prob, vertices_prob in prob_maps_vertices.items():
            # if vtx from cortex is in vertices of prob map, add name to list of labels with vtx, add stat value
            match_idx = np.where(vertices_prob == vtx)
            # if vertex is in probability map
            if match_idx[0].shape[0] > 0:
                # get vertex, RAS, and stat values for the given vertex that is in prob map
                vertices_prob_idx = match_idx[0][0]
                labels_with_vtx = np.append(labels_with_vtx, label_prob)
                vertices_with_vtx = np.append(vertices_with_vtx, vertices_prob[vertices_prob_idx])
                RAS_with_vtx = np.concatenate((RAS_with_vtx, np.reshape(prob_maps_RAS[label_prob][vertices_prob_idx,:],(1,3))),axis=0)
                stat_with_vtx = np.append(stat_with_vtx, prob_maps_stat[label_prob][vertices_prob_idx])

        # if vertex was in at least one prob map, get the max value and add to MPM file
            # also required to have a probability of 0.33 or higher
        if (labels_with_vtx.shape[0] > 0):

            if (np.max(stat_with_vtx) > 1/3):

                max_idx = np.argmax(stat_with_vtx)

                label_max = labels_with_vtx[max_idx]
                RAS_max = RAS_with_vtx[max_idx,:]
                stat_max = stat_with_vtx[max_idx]

                MPM_vertices[label_max] = np.append(MPM_vertices[label_max], vtx)
                MPM_RAS[label_max] = np.concatenate((MPM_RAS[label_max], np.reshape(RAS_max,(1,3))),axis=0)
                MPM_stat[label_max] = np.append(MPM_stat[label_max], stat_max)

    # save out each entry MPM as a separate label file in Freesurfer

    for i, middle_frontal_label in enumerate(middle_frontal_label_names):

        MPM_path = fsaverage_space_labels + 'prob_maps/{}/lh.{}_PROB_MPM_{}.label'.format(left_out_sub, project_id, middle_frontal_label)
        try:
            # make probabilistic label array for albel file
            prob_array = np.zeros(shape=(MPM_vertices[middle_frontal_label].shape[0],5),dtype=float)
            prob_array[:,0] = MPM_vertices[middle_frontal_label]
            prob_array[:,1:4] = MPM_RAS[middle_frontal_label]
            prob_array[:,-1] = MPM_stat[middle_frontal_label]
            np.savetxt(MPM_path, prob_array, fmt='%-2d  %2.3f  %2.3f  %2.3f %1.10f')

            # edit first two lines of label file to match Freesurfer
            f = open(MPM_path, 'r')
            edit_f = f.read()
            f.close()
            f = open(MPM_path, 'w')
            f.write('#!ascii label  , from subject fsaverage vox2ras=TkReg\n{}\n'.format(MPM_vertices[middle_frontal_label].shape[0]))
            f.write(edit_f)
            f.close()
        except Exception:
            pass
    print('Left Hemisphere PROB MPMs written for ', left_out_sub,' \n\n\n') 


    # loop through labels, load prob map and make empty values in MPM dicts
    for i, middle_frontal_label in enumerate(middle_frontal_label_names):
        try:
            #load the prob mpa for the given label
            vertices, RAS, stat = read_label_stat(fsaverage_space_labels + 'prob_maps/{}/lh.{}_PROB_{}.label'.format(left_out_sub, project_id, middle_frontal_label))
            prob_maps_vertices[middle_frontal_label] = vertices
            prob_maps_RAS[middle_frontal_label] = RAS
            prob_maps_stat[middle_frontal_label] = stat

            MPM_vertices[middle_frontal_label] = np.empty(0)
            MPM_RAS[middle_frontal_label] =  np.empty(shape=(0,3))
            MPM_stat[middle_frontal_label] = np.empty(0)
        except Exception:
            pass
            #load lh cortex vertices
    vertices_lh, RAS_lh = read_label('/home/weiner/data/fsaverage/label/lh.cortex.label')
        

    # loop through lh cortex vertices
    for vtx in vertices_lh:

        labels_with_vtx = np.empty(0)
        vertices_with_vtx = np.empty(0)
        RAS_with_vtx = np.empty(shape=(0,3))
        stat_with_vtx = np.empty(0)

        for label_prob, vertices_prob in prob_maps_vertices.items():
            # if vtx from cortex is in vertices of prob map, add name to list of labels with vtx, add stat value
            match_idx = np.where(vertices_prob == vtx)
            # if vertex is in probability map
            if match_idx[0].shape[0] > 0:
                # get vertex, RAS, and stat values for the given vertex that is in prob map
                vertices_prob_idx = match_idx[0][0]
                labels_with_vtx = np.append(labels_with_vtx, label_prob)
                vertices_with_vtx = np.append(vertices_with_vtx, vertices_prob[vertices_prob_idx])
                RAS_with_vtx = np.concatenate((RAS_with_vtx, np.reshape(prob_maps_RAS[label_prob][vertices_prob_idx,:],(1,3))),axis=0)
                stat_with_vtx = np.append(stat_with_vtx, prob_maps_stat[label_prob][vertices_prob_idx])

        # if vertex was in at least one prob map, get the max value and add to MPM file
            # also required to have a probability of 0.33 or higher
        if (labels_with_vtx.shape[0] > 0):

            if (np.max(stat_with_vtx) > 1/3):

                max_idx = np.argmax(stat_with_vtx)

                label_max = labels_with_vtx[max_idx]
                RAS_max = RAS_with_vtx[max_idx,:]
                stat_max = stat_with_vtx[max_idx]

                MPM_vertices[label_max] = np.append(MPM_vertices[label_max], vtx)
                MPM_RAS[label_max] = np.concatenate((MPM_RAS[label_max], np.reshape(RAS_max,(1,3))),axis=0)
                MPM_stat[label_max] = np.append(MPM_stat[label_max], stat_max)

    # save out each entry MPM as a separate label file in Freesurfer

    for i, middle_frontal_label in enumerate(middle_frontal_label_names):

        MPM_path = fsaverage_space_labels + 'prob_maps/{}/lh.{}_PROB_MPM_{}.label'.format(left_out_sub, project_id, middle_frontal_label)
        try:
            # make probabilistic label array for albel file
            prob_array = np.zeros(shape=(MPM_vertices[middle_frontal_label].shape[0],5),dtype=float)
            prob_array[:,0] = MPM_vertices[middle_frontal_label]
            prob_array[:,1:4] = MPM_RAS[middle_frontal_label]
            prob_array[:,-1] = MPM_stat[middle_frontal_label]
            np.savetxt(MPM_path, prob_array, fmt='%-2d  %2.3f  %2.3f  %2.3f %1.10f')

            # edit first two lines of label file to match Freesurfer
            f = open(MPM_path, 'r')
            edit_f = f.read()
            f.close()
            f = open(MPM_path, 'w')
            f.write('#!ascii label  , from subject fsaverage vox2ras=TkReg\n{}\n'.format(MPM_vertices[middle_frontal_label].shape[0]))
            f.write(edit_f)
            f.close()
        except Exception:
            pass
    print('Left Hemisphere PROB MPMs written for ', left_out_sub,' \n\n\n') 

    # save out each entry MPM as a separate binary label file in Freesurfer

    for i, middle_frontal_label in enumerate(middle_frontal_label_names):

        MPM_path = fsaverage_space_labels + 'prob_maps/{}/lh.{}_PROB_MPM_binary_{}.label'.format(left_out_sub, project_id ,middle_frontal_label)
        try:
            # make probabilistic label array for albel file
            prob_array = np.zeros(shape=(MPM_vertices[middle_frontal_label].shape[0],5),dtype=float)
            prob_array[:,0] = MPM_vertices[middle_frontal_label]
            prob_array[:,1:4] = MPM_RAS[middle_frontal_label]
            prob_array[:,-1] = 1
            np.savetxt(MPM_path, prob_array, fmt='%-2d  %2.3f  %2.3f  %2.3f %1.10f')

            # edit first two lines of label file to match Freesurfer
            f = open(MPM_path, 'r')
            edit_f = f.read()
            f.close()
            f = open(MPM_path, 'w')
            f.write('#!ascii label  , from subject fsaverage vox2ras=TkReg\n{}\n'.format(MPM_vertices[middle_frontal_label].shape[0]))
            f.write(edit_f)
            f.close()

        except Exception:
            pass
        
    print('Left Hemisphere PROB Binary MPMs written for ', left_out_sub, '\n\n\n')


