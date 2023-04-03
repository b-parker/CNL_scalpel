


def main():
    subjects_dir = '/Users/benparker/Desktop/cnl/neurocluster/HCP/subjects'
    subjects_list_path = f"{subjects_dir}/subject_lists/HCP_processed_subs.txt"
    subject_list = get_subjects_list(subjects_list=subjects_list_path,
                                     subjects_dir=subjects_dir)
    
    
    sulci_list = ['MCGS', 'POS', 'prculs', 'prcus1', 'prcus2', 'prcus3', 'sbps', '2', '1', '3', 'w', 'x', 'y', 'isms']
    

    sorted_sulci_dict = sort_subjects_and_sulci(subject_list, sulci_list=sulci_list)

    create_freesurfer_ctab('test_annot', sulci_list, subjects_dir)

    dict_to_JSON(sorted_sulci_dict, '/Users/benparker/Desktop/cnl/subjects', 'test_annot')
    json_filename = f"{subjects_dir}test_annot.json"

    create_ctabs_from_dict(subjects_dir, json_filename)

    with open(json_filename) as file:
        sulci_dict = json.load(file)
    
    ctab_project_dir = '/Users/benparker/Desktop/cnl/subjects/'

    for subject_path in subject_list:
        subject = os.path.basename(subject_path)
        for hemi in ['lh', 'rh']:
            sulcus_list = sulci_dict[f"{hemi}_{subject}"]
            ctab_sulci = '_'.join(sulcus_list)
            ctab_path = f"{ctab_project_dir}/{ctab_sulci}.ctab"

            freesurfer_label2annot(subjects_dir,
                                   subject_path, 
                                   label_list=sulcus_list,
                                   hemi=hemi,
                                   ctab_path=ctab_path,
                                   annot_name='test_annot'
                                   )

if __name__ == '__main__':
    main()