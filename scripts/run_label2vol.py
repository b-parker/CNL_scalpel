from src.freesurfer_utils import *

def main():
    subjects_dir = "/Users/benparker/Desktop/cnl/subjects"
    annot_name = "test_annot"
    subjects_list = get_subjects_list(subjects_dir = subjects_dir, subjects_list= f"{subjects_dir}/subjects_list.txt")
    hemis = ['lh', 'rh']
    outfile_name = annot_name

    for subject in ['100307']:
        freesurfer_label2vol(subjects_dir = subjects_dir,
                             annot_name = annot_name,
                             subject = subject,
                             hemi = 'lh',
                             outfile_name  = outfile_name)








if __name__ == "__main__":
    main()
