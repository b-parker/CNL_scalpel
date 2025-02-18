from pathlib import Path
import os
from src.utilities.freesurfer_utils import freesurfer_mris_anatomical_stats

def main():
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

    def get_label_paths(subject):
        aparc_label_paths = subjects_dir / subject / 'label' / 'aparc_a2009s_labels'
        dk_label_paths = subjects_dir / subject / 'label' / 'aparc_DKTatlas40_labels'

        aparc_labels = [label for label in aparc_label_paths.iterdir() if label.is_file()]
        dk_labels = [label for label in dk_label_paths.iterdir() if label.is_file()]
        return aparc_labels, dk_labels

    subject_labels = {}

    for subject in subject_list:
        aparc_labels, dk_labels = get_label_paths(subject)
        subject_labels[subject] = {'aparc_labels': aparc_labels, 'dk_labels': dk_labels}

    des_label_names = [subject_labels['100206']['aparc_labels'][label_i].name.split('.')[-2] for label_i in range(len(subject_labels['100206']['aparc_labels']))]
    dk_label_names = [subject_labels['100206']['dk_labels'][label_i].name.split('.')[-2] for label_i in range(len(subject_labels['100206']['dk_labels']))]


    for subject in subject_list[:2]:
        print(f'Beginning mris_anatomical_stats for subject {subject}\n\n')
        for hemi in ['lh', 'rh']:
            for i, label_file in enumerate(subject_labels[subject]['aparc_labels']):
                table_file_folder = subjects_dir / subject / 'label' / 'label_stats' / 'aparc_labels_a2009s_stats'
                table_file_folder.mkdir(parents=True, exist_ok=True)
                table_file = table_file_folder / f'{".".join(label_file.name.split(".")[-3:])}.stats.txt'
                
                print(f'Beginning {i}/{len(subject_labels[subject]["aparc_labels"])}: {hemi} {label_file.name.split(".")[-3:]}') 
                mris_anatomical_stats = freesurfer_mris_anatomical_stats(subject_name = subject, hemisphere = hemi, 
                                                            label_file = label_file.absolute(), table_file = table_file.absolute(), subjects_dir = subjects_dir.absolute(), freesurfer_home = freesurfer_home, no_global=True)
     
            for i, label_file in enumerate(subject_labels[subject]['dk_labels']):
                table_file_folder = subjects_dir / subject / 'label' / 'label_stats' / 'aparc_labels_dk_stats'
                table_file_folder.mkdir(parents=True, exist_ok=True)
                table_file = table_file_folder / f'{".".join(label_file.name.split(".")[-3:])}.stats.txt'
                print(f'Beginning {i}/{len(subject_labels[subject]["dk_labels"])}: {hemi} {label_file.name.split(".")[-3:]}')
                mris_anatomical_stats = freesurfer_mris_anatomical_stats(subject_name = subject, hemisphere = hemi, 
                                                            label_file = label_file.absolute(), table_file = table_file.absolute(), subjects_dir = subjects_dir.absolute(), freesurfer_home = freesurfer_home, no_global=True)
    


if __name__ == '__main__':
    main()
