import file_utils
import json
import os

def main():
    subjects_dir = '/home/weiner/HCP/subjects'

    subjects_list_path = "/home/weiner/HCP/subject_lists/HCP_processed_subs_all.txt"

    subject_list = file_utils.get_subjects_list(subjects_list=subjects_list_path,
                                     subjects_dir=subjects_dir)
    project_dir='/home/weiner/HCP/projects/ifrms_HCP/annot_ctab_json/'

    annotation_name = 'WillbrandParker_SciAdv_2022'
    
    sulci_list = ['MCGS', 'POS', 'prculs', 'prcus1', 'prcus2', 'prcus3', 'sbps', 'ifrms', 'sspls_d', 'icgs_p', 'pmcgs', 'sspls_v', 'prculs_v', 'isms']

    ## Create color table

    ##TODO check to see if project_dir/annot_ctab_json exists, create if not

   
    sorted_sulci_dict = file_utils.sort_subjects_and_sulci(subject_list, sulci_list=sulci_list)
    file_utils.create_freesurfer_ctab(ctab_name=annotation_name, label_list=sulci_list, outdir=project_dir)

    # Create json in <project directory> with <annotation_name>.json as filename

    file_utils.dict_to_JSON(dictionary=sorted_sulci_dict, outdir=project_dir, project_name=annotation_name)
    
    json_filename = f"{project_dir}/{annotation_name}.json"

    # Create colortables for all existing sulci combinations; store in <project_dir>
    file_utils.create_ctabs_from_dict(project_colortable_dir=project_dir, json_file=json_filename)

    with open(json_filename) as file:
        sulci_dict = json.load(file)
    

    for subject_path in subject_list:
        subject = os.path.basename(subject_path)
        for hemi in ['lh', 'rh']:
            sulcus_list = sulci_dict[f"{hemi}_{subject}"]
            ctab_sulci = '_'.join(sulcus_list)
            ctab_path = f"{project_dir}/{ctab_sulci}.ctab"

            file_utils.freesurfer_label2annot(subjects_dir,
                                   subject_path, 
                                   label_list=sulcus_list,
                                   hemi=hemi,
                                   ctab_path=ctab_path,
                                   annot_name=annotation_name
                                   )

if __name__ == '__main__':
    main()
