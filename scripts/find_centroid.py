from pathlib import Path
import numpy as np

from src.classes.subject import ScalpelSubject

def main() -> None:
    mount_point = Path('/Users/benparker/Desktop/cnl/neurocluster')
    subjects_dir = mount_point / 'weiner/HCP/subjects'
    project_dir = mount_point / 'weiner/HCP/projects/cortical_viz'
    label_dir =  project_dir / 'prob_maps/ALL_SUBS'

    subject = 'fsaverage'

    labels_raw = list(label_dir.glob('*.label'))
    labels = np.unique([label.stem.split('.')[1] for label in labels_raw])
    
    scal_sub_lh = ScalpelSubject(subject, subjects_dir = subjects_dir, hemi = 'lh')
    scal_sub_rh = ScalpelSubject(subject, subjects_dir = subjects_dir, hemi = 'rh')

    for label in labels:
        scal_sub_lh.load_label(label, custom_label_path=label_dir)
        scal_sub_rh.load_label(label, custom_label_path=label_dir)

        label_lh_centroid_idx, label_lh_centroid_RAS = scal_sub_lh.label_centroid(label, load = True)
        label_rh_centroid_idx, label_rh_centroid_RAS = scal_sub_rh.label_centroid(label, load = True)


        scal_sub_lh.write_label(f'{label}_centroid', custom_label_path = label_dir / 'centroids')
        scal_sub_rh.write_label(f'{label}_centroid', custom_label_path = label_dir / 'centroids')



if __name__ == '__main__':
    main()
