
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

def create_subject_hemi_sulci_idx(annot_idxs):
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

def mask_brains_from_annot(subjects: list, brains: dict, all_sulci: list, annot_idxs: dict, subject_hemi_sulci_idx: dict):
    """ 
    Creates mask volumes from annotations, and saves them to a dictionary of brains

    Parameters
    ----------
    subjects : list - list of subject IDs
    brains : dict - dictionary of brains 
                  - {subject_id : [brain, lh.annot, rh.annot], ...}
    all_sulci : list - list of all possible sulci
    annot_idxs : dict - dictionary of subject_hemi sulci
                      - {'hemi_subjectid' : ['suclus1', 'sulcus2', ...], ...}
                      - from the json file written to annot_ctab_json by fsu.create_ctabs_from_dict
    subject_hemi_sulci_idx : dict - dictionary of subject_hemi sulci index mappings
                      - {'hemi_subjectid' : {'sulcus1' : 0, 'sulcus2' : 1, ...}, ...}
                      - from create_subject_hemi_sulci_idx in this file
    
    Returns
    -------
    brains_updated : dict - dictionary of brains with mask volumes appended
                          - {subject_id : [brain, lh.sulcus1_mask, lh.sulcus2_mask], ...}

    """
    brains_updated = {}
    for subject in subjects:
        
        brains_updated[subject] = [brains[subject][0]]
        # for each hemisphere
        ## loop through all possible sulci, append mask if sulcus exists in subject
        ## or zero volume if not
        for hemi_idx in [1, 2]: 
            if hemi_idx == 1:
                sub_hemi = f"lh_{subject}"
            else:
                sub_hemi = f"rh_{subject}"
            for sulcus  in all_sulci:
                if sulcus in annot_idxs[sub_hemi]:
                    # get index of sulcus as used in subject's annot
                    subject_specific_sulcus_index = subject_hemi_sulci_idx[sub_hemi][sulcus]
                    # Zero any value not equal to subject index
                    new_volume = np.where(brains[subject][hemi_idx] == subject_specific_sulcus_index, brains[subject][hemi_idx], 0)
                    # binarize
                    new_volume_binarized = np.where(new_volume==0, new_volume, 1)
                    # append masked volume to updated brains
                    brains_updated[subject].append(new_volume_binarized)
                else:
                    # else append zero volume
                    zero_volume = np.zeros(brains[subject][0].shape)
                    brains_updated[subject].append(zero_volume)

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


        