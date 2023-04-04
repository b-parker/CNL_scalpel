import os
import glob
import freesurfer_utils

def remove_label(file_path):
    try:
        os.remove(file_path)
        print(f"Removing file at {file_path}")
    except OSError:
        print(f"File does not exist at {file_path}")


def main():
    project_dir = '/home/weiner/HCP'
    subjects_dir = f"{project_dir}/subjects"
    subject_paths = freesurfer_utils.get_subjects_list(f"{project_dir}/subject_lists/HCP_processed_subs_all.txt", subjects_dir)
    for subject_path in subject_paths:
        for hemi in ['lh' , 'rh']:
            label_file = f"{subject_path}/label/{hemi}.WillbrandParker_SciAdv_2022.annot"
            remove_label(label_file)


if __name__ == "__main__":
    main()
