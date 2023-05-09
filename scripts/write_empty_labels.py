"""
This script is used to solve the problem wherein different 
subjects have different conmbinations of labels across PFC, PMC, LPC

Because of this, when we project label2vol, the indexing among the labels is different for each subject.

The solution here is to create labels assigned to 0 vertices for each missing label. This way all subjects
will have the same labels exactly, without changing the label assignments of each voxel in the volume.

Update:
This ... doesn't work. Freesurfer requires label files to have data in them to write to an annotation or project into the volume.


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

    empty_sulci_written = {}

    for idx, subject in enumerate(subjects):
        # Initialize subject
        empty_sulci_written[subject] = []
        for label in sulci_list:
            for hemi in ['lh', 'rh']:
                if os.path.exists(f"{subject_paths[idx]}/label/{hemi}.{label}.label"):
                    pass
                else:
                    # Write label if label path doesn't exist, and add label to subject dict
                    fsu.write_label(label_name=label, label_faces=[], verts=[], hemi=hemi,
                                    subject=subject, subjects_dir=subjects_dir)
                    print(f"Writing empty sulcus at {subject} {hemi} {label}")
                    empty_sulci_written[f"{subject}"].append(f"{hemi}.{label}.label")

    fsu.dict_to_json(dictionary=empty_sulci_written, outdir='/Users/benparker/Desktop', project_name='test_dict')

if __name__ == "__main__":
    main()
                