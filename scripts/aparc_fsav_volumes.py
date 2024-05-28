
from pathlib import Path
import src.utilities.freesurfer_utils

subjects = src.utilities.freesurfer_utils.get_subjects_list(subjects_list='/home/weiner/HCP/subject_lists/processed_HCP_YA_1200.txt', subjects_dir = '/home/connectome-raw')

subjects = [Path(subject).name for subject in subjects]


for subject in subjects:
    subjects_dir = f'/home/connectome-raw/{subject}/T1w'
    src.utilities.freesurfer_utils.freesurfer_label2vol(subjects_dir=subjects_dir,\
                                              subject = subject, \
                                              hemi = 'lh',\
                                              outfile_name = 'aparc_fsav_VTC',
                                              annot_name = 'aparc_fsav_VTC')
    
    src.utilities.freesurfer_utils.freesurfer_label2vol(subjects_dir=subjects_dir,\
                                              subject = subject, \
                                              hemi = 'rh',\
                                              outfile_name = 'aparc_fsav_VTC',
                                              annot_name = 'aparc_fsav_VTC')
