import freesurfer_utils
import json
import os

def main():
    subjects_dir = '/home/weiner/HCP/subjects'

    subjects_list_path = "/home/weiner/HCP/subject_lists/HCP_processed_subs_all.txt"

    subject_list = freesurfer_utils.get_subjects_list(subjects_list=subjects_list_path,
                                     subjects_dir=subjects_dir)
    project_dir='/home/weiner/HCP/projects/ifrms_HCP/annot_ctab_json/'

    annotation_name = 'WillbrandParker_SciAdv_2022'
    
    sulci_list = ['MCGS', 'POS', 'prculs', 'prcus_p', 'prcus_i', 'prcus_a', 'spls', 'ifrms', 'sps', 'sspls_d', 'icgs_p']
 
    ## Create color table

    ##TODO check to see if project_dir/annot_ctab_json exists, create if not

    sulci_colors = {'MCGS': '99 180 193' ,
                    'POS': '128 127 184',
                    'prculs': '159  157 200',
                    'prcus_p': '83  151 62',
                    'prcus_i': '150 83 89',
                    'prcus_a': '190 225 149', 
                    'spls': '134 190 125', 
                    'ifrms': '221 75 57',
                    'sps': '0   66  145',
                    'sspls_d': '159 246 77', 
                    'icgs_p': '174 243 254',}
    
    # Save color table as json in <project directory> with colors_<annotation_name>.json as filename
    freesurfer_utils.dict_to_JSON(dictionary=sulci_colors, outdir=project_dir, project_name=f"colors_{annotation_name}")


    ### Full process
    # sort subject hemispheres by present sulci, stores in dictionary
    sorted_sulci_dict = freesurfer_utils.sort_subjects_and_sulci(subject_list, sulci_list=sulci_list)

    # Create json in <project directory> with <annotation_name>.json as filename

    freesurfer_utils.dict_to_JSON(dictionary=sorted_sulci_dict, outdir=project_dir, project_name=annotation_name)

    json_filename = f"{project_dir}/{annotation_name}.json"

    # Create colortables from that dictionary; store in <project_dir>
    freesurfer_utils.create_ctabs_from_dict(project_colortable_dir=project_dir, json_file=json_filename, palette=sulci_colors)

    with open(json_filename) as file:
        sulci_dict = json.load(file)
    

    for subject_path in subject_list:
        subject = os.path.basename(subject_path)
        for hemi in ['lh', 'rh']:
            sulcus_list = sulci_dict[f"{hemi}_{subject}"]
            ctab_sulci = '_'.join(sulcus_list)
            ctab_path = f"{project_dir}/{ctab_sulci}.ctab"

            freesurfer_utils.freesurfer_label2annot(subjects_dir,
                                   subject_path, 
                                   label_list=sulcus_list,
                                   hemi=hemi,
                                   ctab_path=ctab_path,
                                   annot_name=annotation_name
                                   )

if __name__ == '__main__':
    main()
