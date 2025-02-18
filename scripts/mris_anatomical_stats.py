from pathlib import Path
from src.classes.subject import ScalpelSubject

from src.utilities.freesurfer_utils import freesurfer_mris_anatomical_stats

def main():
    mount_point = Path('/Users/benparker/Desktop/cnl/neurocluster')
    subjects_dir = mount_point / 'weiner' / 'HCP' / 'subjects'
    subject_list_path = mount_point / 'weiner' / 'HCP' / 'subject_lists' / 'HCP_processed_subs_all.txt'
    freesurfer_home = '/Users/benparker/freesurfer'

    with open(subject_list_path, 'r') as f:
        subject_list = f.readlines()

    subject_list = [subject.strip() for subject in subject_list]

    def get_label_paths(subject):
        aparc_label_paths = subjects_dir / subject / 'label' / 'aparc_labels_a2009s'
        dk_label_paths = subjects_dir / subject / 'label' / 'aparc_labels_DK'

        aparc_labels = [label for label in aparc_label_paths.iterdir() if label.is_file()]
        dk_labels = [label for label in dk_label_paths.iterdir() if label.is_file()]
        return aparc_labels, dk_labels

    subject_labels = {}

    for subject in subject_list:
        aparc_labels, dk_labels = get_label_paths(subject)
        subject_labels[subject] = {'aparc_labels': aparc_labels, 'dk_labels': dk_labels}

    des_label_names = [subject_labels['100206']['aparc_labels'][label_i].name.split('.')[-2] for label_i in range(len(subject_labels['100206']['aparc_labels']))]
    dk_label_names = [subject_labels['100206']['dk_labels'][label_i].name.split('.')[-2] for label_i in range(len(subject_labels['100206']['dk_labels']))]


    subject = subject_list[0]
    hemi = 'lh'
    label_file = subject_labels[subject]['aparc_labels'][0]
    subjects_dir_out = Path('/Users/benparker/Desktop/cnl/subjects')
    table_file_folder = subjects_dir / subject / 'label' / 'label_stats' / 'aparc_labels_a2009s_stats'
    table_file_folder.mkdir(parents=True, exist_ok=True)
    table_file = table_file_folder / f'{".".join(label_file.name.split(".")[-3:])}.stats.txt'

    print(table_file)
    mris_anatomical_stats = freesurfer_mris_anatomical_stats(subject_name = subject, hemisphere = hemi, 
                                                            label_file = label_file.absolute(), table_file = table_file.absolute(), subjects_dir = subjects_dir.absolute(), freesurfer_home = freesurfer_home, no_global=True)
    

if __name__ == '__main__':
    main()