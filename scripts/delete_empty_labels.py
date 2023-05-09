"""
Checks if a label file has no vertices.

Deletes file if so

"""


import src.freesurfer_utils as fsu
import numpy as np
import os

def main():

    # subjects_dir = "/home/weiner/HCP/subjects/"
    # ## Get subject paths, and subject names
    # subject_paths =  fsu.get_subjects_list(subjects_list=f"{subjects_dir}/../subject_lists/HCP_processed_subs_all.txt",
    #                                        subjects_dir=subjects_dir)

    subjects_dir = "/Users/benparker/Desktop/cnl/subjects/"
    subject_paths =  fsu.get_subjects_list(subjects_list=f"{subjects_dir}/subjects_list.txt",
                                        subjects_dir=subjects_dir)
    subjects = [i[-6:] for i in subject_paths]

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


    for idx, subject in enumerate(subjects):
        for label in sulci_list:
            for hemi in ['lh', 'rh']:
                sulci_file = f"{subjects_dir}/{subject}/label/{hemi}.{label}.label"
                if os.path.exists(sulci_file):
                    with open(sulci_file, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        if len(lines) < 3:
                            os.remove(sulci_file)
                            print(f"Removed {sulci_file}")
                else: 
                    pass

if __name__ == '__main__':
    main()