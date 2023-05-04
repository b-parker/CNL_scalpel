import src.freesurfer_utils as fsu
import json
import os

def main():
    subjects_dir = '/home/weiner/HCP/subjects'

    subjects_list_path = "/home/weiner/HCP/subject_lists/HCP_processed_subs_all.txt"

    subject_list = fsu.get_subjects_list(subjects_list=subjects_list_path,
                                     subjects_dir=subjects_dir)
    



    project_dir='/home/weiner/HCP/projects/CNL_scalpel/annot_ctab_json/'


    annotation_name = 'PFC_LPC_PMC'
    
    sulci_list = ['MCGS',
                    'POS',
                    'prculs',
                    'prcus_p',
                    'prcus_i',
                    'prcus_a', 
                    'spls', 
                    'ifrms',
                    'sps',
                    'sspls_d', 
                    'icgs_p',
                    'pmcgs',
                    'sspls_v',
                    'prculs_v',
                    'isms'   ,
                    'central',
                    'sprs',
                    'iprs', 
                    'sfs_a', 
                    'sfs_p', 
                    'pmfs_p', 
                    'pmfs_i', 
                    'pmfs_a', 
                    'ifs', 
                    'infs_h', 
                    'infs_v',
                    'painfs_d', 
                    'painfs_v',
                    'ds', 
                    'aalf', 
                    'half', 
                    'ts', 
                    'prts', 
                    'lfms',
                    'IPS', 
                    'IPS-PO', 
                    'SPS', 
                    'aipsJ', 
                    'sB', 
                    'pips', 
                    'iTOS', 
                    'mTOS', 
                    'SmgS', 
                    'STS', 
                    'cSTS1', 
                    'cSTS2', 
                    'cSTS3',
                    'SLOS', 
                    'SLOS2', 
                    'SLOS3', 
                    'SLOS4'
                    ]
    ## Create color table

    # sulci_colors = {'MCGS': '99 180 193' ,
    #                 'POS': '128 127 184',
    #                 'prculs': '159  157 200',
    #                 'prcus_p': '83  151 62',
    #                 'prcus_i': '150 83 89',
    #                 'prcus_a': '190 225 149', 
    #                 'spls': '134 190 125', 
    #                 'ifrms': '221 75 57',
    #                 'sps': '0   66  145',
    #                 'sspls_d': '159 246 77', 
    #                 'icgs_p': '174 243 254',
    #                 'pmcgs': '204 157 66',
    #                 'sspls_v': '255 118 32',
    #                 'prculs_v': '255 118 104',
    #                 'isms': '33 224 104'}
    
    # Save color table as json in <project directory> with colors_<annotation_name>.json as filename
    #freesurfer_utils.dict_to_JSON(dictionary=sulci_colors, outdir=project_dir, project_name=f"colors_{annotation_name}")


    ### IDEAS
    ## - Record the string of all unique sulci in a dictionary, key Unique File Id value exact sulci
    ## - Store a filename as the last elements / other list item in the hemi_subject json
    ## 
    ##
    ##
    ##

    ### Full process
    # sort subject hemispheres by present sulci, stores in dictionary
    sorted_sulci_dict = fsu.sort_subjects_and_sulci(subject_list, sulci_list=sulci_list)

    # Create json in <project directory> with <annotation_name>.json as filename

    fsu.dict_to_JSON(dictionary=sorted_sulci_dict, outdir=project_dir, project_name=annotation_name)

    sulci_json_filename = f"{project_dir}/{annotation_name}.json"

    ctab_json_filename = f"{project_dir}/{annotation_name}_ctab_files.json"

    # Create colortables from that dictionary; store in <project_dir>
    fsu.create_ctabs_from_dict(project_colortable_dir=project_dir, json_file=sulci_json_filename,sulci_list=sulci_list, project_name=annotation_name)

    with open(sulci_json_filename) as file:
        sulci_dict = json.load(file)

    with open(ctab_json_filename) as file:
        ctab_dict = json.load(file)
    

    with open(ctab_json_filename) as file:
        ctab_dict = json.load(file)

    
        ## Edit filepath to reference the newly created ctab dictionary
    for subject_path in subject_list:
        subject = os.path.basename(subject_path)
        for hemi in ['lh', 'rh']:
            sulcus_list = sulci_dict[f"{hemi}_{subject}"]
            for key, value in ctab_dict.items():
                if value == sulcus_list:
                    ctab_path = f"{project_dir}/{key}.ctab"

            fsu.freesurfer_label2annot(subjects_dir,
                                   subject_path, 
                                   label_list=sulcus_list,
                                   hemi=hemi,
                                   ctab_path=ctab_path,
                                   annot_name=annotation_name
                                   )

if __name__ == '__main__':
    main()
