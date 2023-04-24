from ..freesurfer_utils import *

def main():
    subjects_dir = "/home/weiner/HCP/subjects"
    annot_name = "WillbrandParker_SciAdv_2022"
    subjects_list = get_subjects_list("/home/weiner/HCP/subject_lists/HCP_processed_subs_all.txt")
    hemis = ['lh', 'rh']
    outfile_name = annot_name

    for subjects in ['100307']:
        freesurfer_label2vol(subjects_dir = subjects_dir,
                             annot_name = annot_name,
                             subject = subject,
                             hemi = 'lh',
                             outfile_name  = outfile_name)








if __name__ == "__main__":
    main()
