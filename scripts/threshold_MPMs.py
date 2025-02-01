from pathlib import Path
from src.classes.subject import ScalpelSubject
import numpy as np

def main():

    print('Thresholding MPMs')
    mount_dir = Path('/Users/benparker/Desktop/cnl/neurocluster')
    project_dir = mount_dir / 'weiner' / 'HCP' / 'projects' / 'cortical_viz'
    subjects_dir = mount_dir / 'weiner' / 'HCP' / 'subjects'

    labels_filename = mount_dir / 'weiner' / 'bparker' / 'code' / 'misc_utils' / 'files_for_thresholding.txt'

    print(f'Loading labels from {labels_filename}')
    with open(labels_filename, 'r') as f:
        labels = f.readlines()

    thresholds = [0.1, .15, .2]

    thresholds_name = [10, 15, 20]

    for i, threshold in enumerate(thresholds):
        for hemi in ['lh', 'rh']: 
            for label in labels[:2]:

                print(f'Loading {hemi}.{label} from fsaverage')
                sub = ScalpelSubject('fsaverage', hemi = hemi, subjects_dir= subjects_dir)

                label = label.strip()
                label_dir = project_dir / 'prob_maps' / 'ALL_SUBS'
                save_dir = label_dir / 'thresholded'
                save_dir.mkdir(exist_ok=True)

                full_label_name = f'MPM_all_subjects_incl_PROB_{label}'
                
                
                sub.load_label(f'{full_label_name}', custom_label_path=label_dir, include_value= True)

                print(f'Loaded {label} from {label_dir}')
                thresholded_val = np.argwhere(sub.labels[full_label_name]['value'] > threshold)
                
                thresholded_idxs = sub.labels[full_label_name]['idxs'][thresholded_val]
                thresholded_ras = sub.labels[full_label_name]['RAS'][thresholded_val]

                sub.remove_label(full_label_name)

            
                if len(thresholded_idxs) == 0:
                    print(f'No vertices above threshold for {label} at {threshold}')
                    continue
                else:
                    print(f'Found {len(thresholded_idxs)} vertices above threshold for {label} at {threshold}')
                    sub.load_label(f'{label}_{thresholds_name[i]}', label_idxs=thresholded_idxs, label_RAS=thresholded_ras, custom_label_path=save_dir)

                    print(f'Saving {label}_{threshold} to {save_dir}') 
                    custom_label_path = save_dir 
                    sub.write_label(f'{label}_{thresholds_name[i]}', custom_label_path=custom_label_path)
                    print(f'Saved {label}_{threshold} to {custom_label_path} \n')
                    

if __name__ == '__main__':
    main()