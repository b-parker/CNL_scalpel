import src.freesurfer_utils


def main():
    subjects_dir = '/home/weiner/DevProso/subjects'   
    subjects_list = '/home/weiner/DevProso/subjects/all_subjects_list.txt'
    sulci_dict = {'painfs_d' : 'pimfs_d', 'painfs_v' : 'pimfs_v', 'infs_h' : 'imfs_h', 'infs_v' : 'imfs_v', 
                  'iTOS' :'lTOS', 'SLOS' :'slocs_v', 'SLOS2': 'slocs_d', 'SLOS3':'pAngs_v', 'SLOS4':'pAngs_d'}
   
    src.freesurfer_utils.rename_labels(subjects_dir, subjects_list, sulci_dict, by_copy=True)
   
if __name__ == '__main__':
    main()

