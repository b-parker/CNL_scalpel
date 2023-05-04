import src.freesurfer_utils as fsu
import os

def main():
    project_dir = "/home/weiner/bparker/data"
    tarfile_name = "rh_PFC_LPC_PMC"
    subject_list = "/home/weiner/HCP/subject_lists/HCP_processed_subs_all.txt"
    subjects_dir = "/home/weiner/HCP/subjects"

    fsu.create_tar_for_file_from_subject_list(project_dir = project_dir,
                                 tarfile_name = tarfile_name,
                                 subject_list = subject_list,
                                              subjects_dir = subjects_dir,
                                 filepath_from_subject_dir = '/mri/rh.PFC_LPC_PMC.nii.gz')

if __name__ == "__main__":
    main()
    
