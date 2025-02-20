from src.utilities.freesurfer_utils import freesurfer_mris_label2annot, freesurfer_mris_label2label, freesurfer_mris_anatomical_stats, freesurfer_mris_annotation2label
from pathlib import Path
import os


def main():
    # Set local to True if running on local machine
    local = False
    
    if local:
        mount_point = Path('/Users/benparker/Desktop/cnl/neurocluster')
        freesurfer_home = '/Users/benparker/freesurfer'
    else:
        mount_point = Path('/home')
        freesurfer_home = os.environ.get('FREESURFER_HOME')
        
    subjects_dir = mount_point / 'weiner' / 'HCP' / 'subjects'
    subject_list_path = mount_point / 'weiner' / 'HCP' / 'subject_lists' / 'HCP_processed_subs_all.txt'


    with open(subject_list_path, 'r') as f:
        subject_list = f.readlines()

    subject_list = [subject.strip() for subject in subject_list]

    fsaverage_path = subjects_dir / 'fsaverage'

    yeo_7 = 'Yeo2011_7Networks_N1000'
    yeo_17 = 'Yeo2011_17Networks_N1000'

    # make directory for labels
    yeo_7_label_path = fsaverage_path / 'label' / 'yeo_7'
    yeo_7_label_path.mkdir(parents=True, exist_ok=True)
    yeo_17_label_path = fsaverage_path / 'label' / 'yeo_17'
    yeo_17_label_path.mkdir(parents=True, exist_ok=True)


    ## annot2label of the yeo parcels
    freesurfer_mris_annotation2label(subject_id = 'fsaverage', subject_dir = subjects_dir, freesurfer_home = freesurfer_home,
                                     annot_name = yeo_7, hemi = 'lh', label_dir = yeo_7_label_path)
    freesurfer_mris_annotation2label(subject_id = 'fsaverage', subject_dir = subjects_dir, freesurfer_home = freesurfer_home,
                                        annot_name = yeo_7, hemi = 'rh', label_dir = yeo_7_label_path)
    
    freesurfer_mris_annotation2label(subject_id = 'fsaverage', subject_dir = subjects_dir, freesurfer_home = freesurfer_home,
                                        annot_name = yeo_17, hemi = 'lh', label_dir = yeo_17_label_path)
    freesurfer_mris_annotation2label(subject_id = 'fsaverage', subject_dir = subjects_dir, freesurfer_home = freesurfer_home,
                                        annot_name = yeo_17, hemi = 'rh', label_dir = yeo_17_label_path)
    

    # label2label of the yeo labels into all subjects
    yeo_7_labels = [label for label in yeo_7_label_path.iterdir() if label.is_file()]
    yeo_17_labels = [label for label in yeo_17_label_path.iterdir() if label.is_file()]

    for subject in subject_list:
        for label in yeo_7_labels:
            freesurfer_mris_label2label(src_subject = 'fsaverage', src_label = label, 
                                        trg_subject = subject, trg_label = label, reg_method = 'surface',
                                        hemi = ['lh', 'rh'], subjects_dir = subjects_dir, freesurfer_home = freesurfer_home)
            
        for label in yeo_17_labels:
            freesurfer_mris_label2label(src_subject = 'fsaverage', src_label = label, 
                                        trg_subject = subject, trg_label = label, reg_method = 'surface',
                                        hemi = ['lh', 'rh'], subjects_dir = subjects_dir, freesurfer_home = freesurfer_home)

    # mris_anatomical_stats of the yeo labels in all subjects / glasser labels
    glasser_label_path_ex = subjects_dir / '100206' / 'label' / 'glasser'
    glasser_labels = [label for label in glasser_label_path_ex.iterdir() if label.is_file()]

    all_labels = [yeo_7_labels, yeo_17_labels, glasser_labels]
    label_stats_name = ['yeo_7_label_stats', 'yeo_17_label_stats', 'glasser_label_stats']

    for subject in subject_list:
        for i, label_group in enumerate(all_labels):
            label_stats_path = subjects_dir / subject / 'label' / 'label_stats' / label_stats_name[i]
            label_stats_path.mkdir(parents=True, exist_ok=True)
            for label in label_group:
                table_file = label_stats_path / f'{label.name}.stats.txt'
                freesurfer_mris_anatomical_stats(subject_name = subject, hemisphere = label.name.split('.')[-2],
                                                label_file = label, table_file = table_file, subjects_dir = subjects_dir,
                                                freesurfer_home = freesurfer_home, no_global = True)
                

## label2annot of subject yeo labels into one annot / glasser labels
        annot_names = [yeo_7, yeo_17, 'glasser']

    
        for i, label_group in enumerate(all_labels):
            annot_path = subjects_dir / subject/ 'label' / annot_names[i]

            freesurfer_mris_label2annot(subject_id = subject, subject_dir = subjects_dir, freesurfer_home = freesurfer_home,
                                        label_dir = label_group, annot_name = annot_names[i], hemi = ['lh', 'rh'], annot_dir = annot_path)
