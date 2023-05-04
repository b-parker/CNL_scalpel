import src.freesurfer_utils as fsu
import os
def main():
    subjects_dir = "/home/weiner/HCP/subjects"
    annot_name = "PFC_LPC_PMC"
    subjects_list = fsu.get_subjects_list(subjects_dir = subjects_dir, subjects_list="/home/weiner/HCP/subject_lists/HCP_processed_subs_all.txt")
    hemis = ['lh', 'rh']
    outfile_name = annot_name

    
    for subject_file in subjects_list:
        for hemi in hemis:
            subject = os.path.basename(subject_file)
            fsu.freesurfer_label2vol(subjects_dir = subjects_dir,
                             annot_name = annot_name,
                             subject = subject,
                             hemi = hemi,
                             outfile_name  = outfile_name)








if __name__ == "__main__":
    main()
