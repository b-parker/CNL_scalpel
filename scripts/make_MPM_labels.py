import os
import pdb
import src.utilities.freesurfer_utils as fsu
from pathlib import Path
from itertools import product
import logging


def main():
    mount_point = "/Users/benparker/Desktop/cnl/neurocluster"
    subjects_directory = f"{mount_point}/weiner/HCP/subjects"
    target_directory = f"{mount_point}/weiner/HCP/projects/cortical_viz/fsavg_projected_labels"
    subject_list_txt = f"{mount_point}/weiner/HCP/subject_lists/HCP_processed_subs_all.txt"

    logging.basicConfig(filename='/Users/benparker/Desktop/MPM_label_creation.log', level=logging.INFO, format='%(asctime)s %(message)s')

    label_list = ['MCGS',
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

    hemis = ['lh', 'rh']

    subject_list = fsu.get_subjects_list( subject_list_txt, subjects_directory)
    
    # Project to fsaverage all subject-space labels 
    for label_hemi in list(product(label_list, hemis)):
        for subject in subject_list:

            # Ensure target directory exists
            subject = Path(subject).stem
            label = label_hemi[0]
            hemi = label_hemi[1]
            os.makedirs(f"{target_directory}/{subject}", exist_ok=True)
            subject_target_directory = f"{target_directory}/{subject}"
            
            # Check if label already exists
            if not os.path.exists(f"{subject_target_directory}/{subject}.{hemi}.{label}.label"): 
                
                # If no label exists, project label to fsaverage
                try:
                    fsu.freesurfer_label2label(source_subjects_dir=subjects_directory, source_subject=subject,
                                            target_subject_dir=subjects_directory, target_subject='fsaverage',
                                            source_label = label,
                                            target_label_dir= subject_target_directory, target_label=label,
                                            hemi = hemi)
                except:
                    print(f"Label {hemi}.{label}.label not found for subject {subject}")
                    continue
            else:
                logging.info(f"[To FSAVERAGE] Label {label} {hemi} exists for {subject}")
                pass
            
            # Create probability maps
            # <done with MPM_labels.py in lab scripts github repostitory

            # Project MPM maps back into native space
            # Check if label already exists
            subject_target_directory = f"{subjects_directory}/{subject}/label/cortical_viz/MPM"
            if not os.path.exists(f"{subject_target_directory}/{subject}.{hemi}.MPM_proj_label.{label}.label"):
                # If no label exists, project label to fsaverage
                # pdb.set_trace()
                try:
                    fsu.freesurfer_label2label(source_subjects_dir=subjects_directory, source_subject=subject,
                                            target_subject_dir=subjects_directory, target_subject=subject,
                                            source_label_dir= f"{mount_point}/weiner/HCP/projects/cortical_viz/fsavg_projected_labels/{subject}/", source_label = label,
                                            target_label_dir= f"{subject_target_directory}", target_label=f"MPM_proj_label.{label}",
                                            hemi = hemi)
                except:
                    print(f"Label {hemi}.{label}.label not found for subject {subject}")
                    # create log
                    logging.info(f"Label {hemi}.{label}.label not found for subject {subject}")
                    continue
                    

            else:
                logging.info(f"[TO NATIVE] Label {label} {hemi} exists for {subject}")
                continue
    


    # # Threshold probabilty maps by maximum probability

    # txt_file_loc = "/Users/benparker/Desktop/sulci_names.txt"
    # with open(txt_file_loc, 'w') as f:
    #     for label in label_list:
    #         f.write(f"{label}\n")
if __name__ == '__main__':
    main()