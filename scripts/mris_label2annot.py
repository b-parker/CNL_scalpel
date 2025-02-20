
from src.utilities.freesurfer_utils import freesurfer_mris_label2annot
from pathlib import Path
import os


def main():
    local = True
    
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

    subject_labels = {}

    kong_labels = ['Default_network', 'Control_network', 'Dorsal_Attention_network','Visual_network', 'Ventral_Attention_network', 'Somato_Motor_network', 'TemporalParietal']

    for subject in subject_list:
        subject_label_path = subjects_dir / subject / 'label'
        subject_labels[subject] = []
        for hemi in ['lh', 'rh']:
           for label in kong_labels:
                label_file = subject_label_path / f'{hemi}.{label}.label'
                if label_file.exists():
                    subject_labels[subject].append(label_file)
                else:
                    print(f'{hemi} {label_file} does not exist for subject {subject}')
    
                     




if __name__ == '__main__':
    main()