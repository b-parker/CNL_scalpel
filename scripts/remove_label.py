import os
import glob
import freesurfer_utils

def remove_label(file_path):
    assert os.path.exists(file_path), "File does not exist: {}".format(file_path)
    print(f"Removing {file_path}")
    os.remove(file_path)


def main():
    project_dir = '/home/weiner/HCP'
    subject_paths = freesurfer_utils.get_subjects_list(f"{project_dir}/subject_lists/HCP_processed_subs.txt")
    for subject_path in subject_paths:
        label_files = glob.glob(f"{subject_path}/label/?h.WillbrandParker_SciAdv_2022.annot")
        for label_file in label_files:
            remove_label(label_file)


if __name__ == "__main__":
    main()