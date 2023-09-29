from pathlib import Path
from src import freesurfer_utils



subjects = freesurfer_utils.get_subjects_list(subjects_list='/home/weiner/HCP/subject_lists/processed_HCP_YA_1200.txt', subjects_dir = '/home/connectome-raw')

subjects = [Path(subject).name for subject in subjects]


label_list = ['CoS_sulcus', 'OTS_sulcus', 'MFS_sulcus', 'MFSfs_sulcus']

for hemi in ['lh', 'rh']:
    for subject in ['100307']:
        subjects_dir = f'/home/connectome-raw/{subject}/T1w'
        subject_path = f'/home/weiner/HCP/projects/CNL_scalpel/HCP_YA_1200/{subject}'
        ctab_path = '/home/weiner/HCP/projects/CNL_scalpel/aparc_fsav_VTC.ctab'
    
        freesurfer_utils.freesurfer_label2annot(subjects_dir=subjects_dir, subject_path=subject_path, label_list=label_list, hemi=hemi, ctab_path=ctab_path, annot_name='aparc_fsav_VTC')