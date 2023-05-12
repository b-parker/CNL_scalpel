
import src.freesurfer_utils as src
import json
from pathlib import Path
import numpy as np
import nibabel as nb

def load_brains_and_annots(subjects_path_list : list, annot_name: str) -> dict:
    """ 
    Loads brain.mgz and {annot}.nii.gz from cluster, saves them under subject ID in a dict.
    The dictionary is structured as {subject_id : [brain, lh.annot, rh.annot]}

    Parameters
    ----------
    subjects_path_list : list - list of subject paths
    annot_name : str - annotation name

    Returns
    -------
    brains : dict - {subject_id : [brain, lh.annot, rh.annot], ...}
    """
    brains = {}
    
    subjects = [Path(subject_path).name for subject_path in subjects_path_list]
    for idx, subject_path in enumerate(subjects_path_list):
        brains[subjects[idx]] = [nb.load(f"{subject_path}/mri/brain.mgz").get_fdata()]
        brains[subjects[idx]].append(nb.load(f"{subject_path}/mri/lh.{annot_name}.nii.gz").get_fdata())
        brains[subjects[idx]].append(nb.load(f"{subject_path}/mri/rh.{annot_name}.nii.gz").get_fdata())
    
    return brains

def load_subject_hemi_sulci_idx(annot_idxs):
    """ 
    Create dict of subject_hemi sulci index mappings for update_sulcus_indices

    Parameters
    ----------
    annot_idxs : dict - dictionary of subject_hemi annotations 
                      - of the form {'hemi_subjectid' : ['suclus1', 'sulcus2', ...], ...}
                      - outputted into annot_ctab_json by fsu.create_ctabs_from_dict
    """
    sub_hemi_sulci_idx = {}
    for subject_hemi in annot_idxs.keys():
        sub_hemi_sulci_idx[subject_hemi] = {}
        for i, sulcus in enumerate(annot_idxs[subject_hemi]):
            sub_hemi_sulci_idx[subject_hemi][sulcus] = i
    
    return sub_hemi_sulci_idx


def update_sulcus_indices(annot_idxs : dict, sub_hemi_sulci_idx : dict, all_sulci_idx : dict, brains : dict):
    """ 
    Updates the sulcus indices in an annotation volume to match the sulcus indices in all other volumes.

    Detail:
    When we project label2vol of an annotation with different numbers of sulci across subjects, the same sulci will have different
    indices across subject hemispheres. i.e. if we plan to project all PFC sulci, and some subjects have 12 sulci and others have
    15, then the pmfs-p may not be the same across subjects. 

    To solve this we need to map all sulci indices in each subject to the correct index from the full list of possible sulci.

    Parameters
    ----------
    annot_idxs : dict - dictionary of subject_hemi annotations 
                      - of the form {'hemi_subjectid' : ['suclus1', 'sulcus2', ...], ...}
                      - json written to annot_ctab_json by fsu.create_ctabs_from_dict
                      - dict made with load_subject_hemi_sulcus
    sub_hemi_sulci_idx : dict of dict - maps the sulci in each annot_idx key to an index
                      - of the form {'subject_hemi' : {'sulcus' : index}}
    all_sulci_idx : dict - maps all sulci to correct indexes
                      - of the form {'sulcus' : index}
    brains : dict - dict of all volumes outputted from load_brains_and_annots
                      - of the form {'subject_id', [np.array('brain.mgz'), np.array('lh.annot'), np.array('rh.annot')]}

    Returns
    -------
    brains_updated : dict - dict of all volumes outputted from load_brains_and_annots with updated sulcus indices
                          - of the form {'subject_id', [np.array('brain.mgz'), np.array('lh.annot'), np.array('rh.annot')]}
    """
    brains_updated = np.copy(brains)

    for subject_hemi in annot_idxs.keys():
        subject = subject_hemi.split('_')[1]
        hemi = subject_hemi.split('_')[0]
        
        incorrect_indexes = sub_hemi_sulci_idx[subject_hemi]
        correct_indexes = all_sulci_idx
        
        if hemi == 'lh':
            old_brain = brains_updated[subject][1]
            new_brain = np.copy(old_brain)
            for sulcus in incorrect_indexes.keys():
                old_sulcus_idx = incorrect_indexes[sulcus]
                new_sulcus_idx = correct_indexes[sulcus]
                if old_sulcus_idx != new_sulcus_idx: 
                    print(f"{subject} {hemi} is different for {sulcus}, changing {old_sulcus_idx} to {new_sulcus_idx}")
                    new_brain[old_brain == old_sulcus_idx] = new_sulcus_idx

                    brains_updated[subject][1] = new_brain
        
        if hemi == 'rh':
            old_brain = brains_updated[subject][2]
            new_brain = np.copy(old_brain)
            for sulcus in incorrect_indexes.keys():
                old_sulcus_idx = incorrect_indexes[sulcus]
                new_sulcus_idx = all_sulci_idx[sulcus]
                if old_sulcus_idx != new_sulcus_idx: 
                    print(f"{subject} {hemi} is different for {sulcus}, changing {old_sulcus_idx} to {new_sulcus_idx}")
                    new_brain[old_brain == old_sulcus_idx] = new_sulcus_idx

                    brains_updated[subject][2] = new_brain
    return brains_updated


def create_dict_from_list_by_index(original_list):
    """  
    Takes an original_list, and then creates a dictionary where each
    list element is a key, and its index in the original dictionary is its value

    Parameters
    ----------
    original_list : list

    Returns
    -------
    list_indexes_dict : dict
    """

    list_indexes_dict = {}
    for i, element in enumerate(original_list):
        list_indexes_dict[element] = i
    
    return list_indexes_dict


        