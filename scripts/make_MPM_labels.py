import os
import src.utilities.freesurfer_utils as fsu
from pathlib import Path
from itertools import product


def main():
    subjects_directory = "/Users/benparker/Desktop/cnl/neurocluster/weiner/HCP/subjects"
    target_directory = "/Users/benparker/Desktop/cnl/neurocluster/weiner/HCP/projects/cortical_viz/fsavg_projected_labels"
    subject_list_txt = "/Users/benparker/Desktop/cnl/neurocluster/weiner/HCP/subject_lists/HCP_processed_subs_all.txt"

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
                except FileNotFoundError:
                    print(f"Label {hemi}.{label}.label not found for subject {subject}")
            else:
                continue
    # Create probability maps



    # Threshold probabilty maps by maximum probability


if __name__ == '__main__':
    main()