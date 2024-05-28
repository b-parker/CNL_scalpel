import src.preprocessing_volumes as ppv
import src.utilities.freesurfer_utils as fsu
import numpy as np
import nibabel as nb
import json
import os
from pathlib import Path
import h5py
import gc

def main():
    subjects_dir = "/Users/benparker/Desktop/cnl/neurocluster/HCP/subjects"
    subjects_path_list = fsu.get_subjects_list(subjects_dir=subjects_dir, 
                                           subjects_list="/Users/benparker/Desktop/cnl/neurocluster/HCP/subject_lists/HCP_processed_subs_all.txt")    
    batch_size = np.arange(0, len(subjects_path_list), 5)
    batch_size = np.append(batch_size, len(subjects_path_list))

    annot_name = "PFC_LPC_PMC"

    
    sulci_list = ['MCGS',
            'POS',
            'prculs',
            'prcus_p',
            'prcus_i',
            'prcus_a', 
            'spls', 
            'ifrms',
            'sps',
            'sspls_d', 
            'icgs_p',
            'pmcgs',
            'sspls_v',
            'prculs_v',
            'isms'   ,
            'central',
            'sprs',
            'iprs', 
            'sfs_a', 
            'sfs_p', 
            'pmfs_p', 
            'pmfs_i', 
            'pmfs_a', 
            'ifs', 
            'infs_h', 
            'infs_v',
            'painfs_d', 
            'painfs_v',
            'ds', 
            'aalf', 
            'half', 
            'ts', 
            'prts', 
            'lfms',
            'IPS', 
            'IPS-PO', 
            'SPS', 
            'aipsJ', 
            'sB', 
            'pips', 
            'iTOS', 
            'mTOS', 
            'SmgS', 
            'STS', 
            'cSTS1', 
            'cSTS2', 
            'cSTS3',
            'SLOS', 
            'SLOS2', 
            'SLOS3', 
            'SLOS4'
            ]



    annot_idxs_path = "/Users/benparker/Desktop/cnl/neurocluster/HCP/projects/CNL_scalpel/annot_ctab_json/PFC_LPC_PMC.json"
    # annot_idxs_path = "/Users/benparker/Desktop/cnl/subjects/annot_ctab_json/test_annot.json"

    with open(annot_idxs_path, 'r', encoding='utf-8') as annot_idxs:
        annot_idxs = json.load(annot_idxs)
    subject_hemi_sulci_idx = ppv.create_subject_hemi_sulci_idx(annot_idxs)
    

    for batch in batch_size:
        subjects = [Path(subject_path).name for subject_path in subjects_path_list][batch-5:batch]

        brains = ppv.load_brains_and_annots(subjects_path_list[batch-5:batch], annot_name=annot_name)
        brains_masked = ppv.mask_brains_from_annot(subjects=subjects, brains=brains, all_sulci=sulci_list, 
                                           annot_idxs=annot_idxs, subject_hemi_sulci_idx=subject_hemi_sulci_idx)
        

        for subject in subjects:
            save_file = h5py.File(f"/Users/benparker/Desktop/cnl/subjects/{subject}_{annot_name}.h5", 'w')
            print(f"Saving {subject}")
            print("\n")
            stacked_label = np.stack(brains_masked[subject][:-1], axis=0)
            save_file.create_dataset('label', 
                                    stacked_label.shape, 
                                    dtype='float64' , 
                                    data = stacked_label,
                                    compression='gzip',
                                    compression_opts=9)
            added_dim = np.expand_dims(brains_masked[subject][-1], 0)
            save_file.create_dataset('raw', 
                                    added_dim.shape, 
                                    dtype='float64' , 
                                    data = added_dim,
                                    compression='gzip',
                                    compression_opts=9)
            

            print(f"{subject} saved")
            print("\n")
            save_file.close()
            gc.collect()


if __name__ == "__main__":
    main()
