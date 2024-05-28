from src.utilities.freesurfer_utils import *

def main():
    subjects_dir = "/home/weiner/HCP/subjects"
    annot_name = "WillbrandParker_SciAdv_2022"
    subjects_list = get_subjects_list("/home/weiner/HCP/subject_lists/HCP_processed_subs_all.txt", subjects_dir)
    hemis = ['lh', 'rh']
    outfile_name = annot_name

    for subject in ['100307']:
        freesurfer_label2vol(subjects_dir = subjects_dir,
                             subject = subject,
                             hemi = 'lh',
                             annot_name = annot_name,
                             outfile_name  = outfile_name)








if __name__ == "__main__":
    main()
