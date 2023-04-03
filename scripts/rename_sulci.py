import file_utils


def main():
    subjects_dir = '/home/weiner/DevProso/subjects'   
    subjects_list = '/home/weiner/DevProso/subjects/all_subjects_list.txt'
    sulci_dict = {'2' : 'ifrms', '1' : 'sspls_d', 'prculs' : 'prculs_d', 'prcus1' : 'prcus_p', 'prcus2' : 'prcus_i', 'prcus3' : 'prcus_a', 'sbps' : 'spls', '3' : 'icgs_p', 'w' : 'pmcgs', 'x' : 'sspls_v', 'y' : 'prculs_v'}

    file_utils.rename_labels(subjects_dir, subjects_list, sulci_dict, by_copy=True)

if __name__ == '__main__':
    main()

