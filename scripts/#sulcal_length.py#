#######################################################################################
#####                           Sulcal Length Script                              #####
#####                                                                             #####
#####  RESULT: A CSV saved to project folder with maximum length within a sulcal  #####
#####  label (calculated in millimeters)                                          #####
#####                                                                             #####
#####  REQUIRMENTS:                                                               #####
#####      1) recon-all on all subjects                                           #####
#####                                                                             #####
#####      2) A subjects list in a .txt file                                      #####
#####                                                                             #####
#####      3) Labels created for sulcus of interest in the format                 #####
#####         <subject_dir>/<subject>/label/<hemi>.<sulcus>.label                 #####
#####         i.e. subjects/1/label/rh.MFS.label                                  #####
#####                                                                             #####
#####      4) All fractionated sulci--sulci with multiple discontinuous sections  #####
#####      -- have been identified and labeled individually. Follow each label    #####
#####      name with a letter i.e. a, b, c. These should be saved in a file       #####
#####      called <manual_label> within each subject's label directory.           #####
#####           i.e. subjects/1/label/manual_label/rh.MFSa.label                  #####
#####                                                                             #####
#####      5) pycortex, nibabel, nilearn must be installed and operational        #####
#####      see download instructions for pycortex here:                           #####
#####          https://gallantlab.org/pycortex/docs/install.html                  #####
#####                                                                             #####
#####                                                                             #####
#####  TO CALL:                                                                   #####
#####    sulcal_length.py <subject_dir> <subject_list.txt> <label_name>           #####
#####    i.e. sulcal_length.py /home/weiner/DevProso/subjects                     #####
#####                          /home/weiner/DevProso/all_subjects_list.txt        #####
#####                          MFS                                                #####                         
#######################################################################################


import cortex
import numpy as np
import nibabel as nib
import nilearn
from nilearn import image, plotting, surface
import scipy
import matplotlib.pyplot as plt 
import os 
import glob
import seaborn as sns
import itertools
import pandas as pd
import sys
import os


############################################################
#####                  Load filepaths                  #####
##### FILL-IN: sub directory, sub list, labels         #####
############################################################

subject_dir = sys.argv[1] 

# generate subject lists 

file = open(sys.argv[2]) 
subs=file.read().splitlines()
file.close
#Define hemis
hemis = [ 'rh', 'lh'] 
#Define labels
labels = [sys.argv[3]] 

#combine hemis and subject names
sub_hemi_combos = list(itertools.product(subs, hemis))


############################################################
##### Only need to run this cell ONCE for each project #####
##### RESULT: pycortex import for each subject         #####
############################################################


for sub in subs:
  cortex.freesurfer.import_subj(sub, freesurfer_subject_dir=subject_dir)



def read_label(label_name):
    """
    Reads a freesurfer-style .label file (5 columns)
    
    Parameters
    ----------
    label_name: str 
    
    Returns 
    -------
    vertices: index of the vertex in the label np.array [n_vertices] 
    RAS_coords: columns are the X,Y,Z RAS coords associated with vertex number in the label, np.array [n_vertices, 3] 
    
    """
    
    # read label file, excluding first two lines of descriptor 
    df_label = pd.read_csv(label_name,skiprows=[0,1],header=None,names=['vertex','x_ras','y_ras','z_ras','stat'],delimiter='\s+')
    
    vertices = np.array(df_label.vertex) 
    RAS_coords = np.empty(shape = (vertices.shape[0], 3))
    RAS_coords[:,0] = df_label.x_ras
    RAS_coords[:,1] = df_label.y_ras
    RAS_coords[:,2] = df_label.z_ras
    
    return vertices, RAS_coords


def max_path_length(sub, hemi, label_filename, path_filename):
    
    """
    Calculates the maximum path length along the surface for a given Freesurfer label, using pycortex 
    (sub must already be saved in pycortex database)
    
    Saves out path_file as 'hemi.label_name.path'
        
    Parameters
    ----------
    sub: str 
    hemi: str ['lh', 'rh']
    label_name: str (to a freesurfer-style .label file (5 columns))
    
    Returns 
    -------
    path_length: float 
    
    """
    print('\n',label_filename)

    # get fiducial and other surfaces from pycortex database 
    surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(sub, 'fiducial')]
    
    label_vertices, label_RAS = read_label(label_filename)
    
    # Going to work with only the left hemisphere
    if hemi == 'lh':
        hem = 0
    elif hemi == 'rh':
        hem = 1
    surf_hem = surfs[hem]
    num_vertices = surfs[hem].pts.shape[0]
    # Get a subsurface object that is only the vertices in this ROI
    vertex_mask = np.zeros(num_vertices, dtype=bool)
    vertex_mask[label_vertices] = True
    subsurf = surf_hem.create_subsurface(vertex_mask=vertex_mask)
    
    # Get which vertices are along the edge of the ROI
    edge_verts = np.nonzero(subsurf.boundary_vertices)[0]
    # For each edge vertex, get distance to the rest of the subsurface
    try:
        dists = np.array([subsurf.geodesic_distance([v]) for v in edge_verts])
        path_length = np.amax(dists)
        print(path_length)
    except RuntimeError: 
        path_length = 'discontinuous'
        print(path_length)
        return path_length
    
### This code is from original notebook and draws a path on the original surface
### Errors due to (likely) update in pycortex
### Retained in case interested

#     # draw a path between the vertices
#     vert_pair = np.where(dists == dists.max())
#     vert1 = edge_verts[vert_pair[0][0]]
#     vert2 = vert_pair[1][0]
    
#     geo_path = np.array(subsurf.geodesic_path(vert1, vert2))
#     # The indices of the vertices you have now are on the subsurface object
#     # But this can easily be inverted to go back to the entire left hemisphere surface
#     #s_vert1 = subsurf.subsurface_vertex_inverse[vert1]
#     #s_vert2 = subsurf.subsurface_vertex_inverse[vert2]
#     # And you can do the exact same for the path
#     s_geo_path = subsurf.subsurface_vertex_inverse[geo_path]
    
#     # make geo path into a path readable by freesurfer
#     indices_path = np.arange(label_vertices.shape[0])[np.in1d(label_vertices, s_geo_path)]
        
#     if s_geo_path.shape[0] > indices_path.shape[0]: 
#         path_length = 'discontinuous'
#         print(path_length)
#         return path_length
#         #break 
#     else:
#         print(path_length)
#         pass
    
#     path_RAS = label_RAS[indices_path]
#     #print(path_RAS.shape)
#     path_vertices = s_geo_path
#     #print(s_geo_path.shape)

#     path_array = np.zeros(shape=(path_vertices.shape[0],4),dtype=float)
#     path_array[:,0:3] = path_RAS
#     path_array[:,3] = path_vertices
#     np.savetxt(path_filename, path_array, fmt='%2.6f %2.6f %2.6f %-2d')
#     # edit first two lines of label file to match Freesurfer
#     f = open(path_filename, 'r')
#     edit_f = f.read()
#     f.close()
#     f = open(path_filename, 'w')
#     f.write('# Path file\nVERSION 2\nBEGINPATH\nNUMPOINTS {}\n'.format(path_vertices.shape[0]))
#     f.write(edit_f)
#     f.close()
#     # edit last line of text file
#     f = open(path_filename, 'r')
#     edit_f = f.read()
#     f.close()
#     f = open(path_filename, 'a')
#     f.write('ENDPATH')
#     #f.write(edit_f)
#     f.close()
    
    return path_length


#################################################################################
#####                                                                       #####
##### Identify any discontinuous sulci and assign them to a list of tuples  #####
##### RESULT: dis_pairs = [('sub', 'hemi'), . . .]                          #####
#####                                                                       #####
##### REQUIREMENT: all discontinuous sulci must be manually identified      #####
##### and saved under subject_dir/label/manual_label directory. To do this, #####
##### create a manual_label directory in each label folder, and save all    #####
##### discontinuous sulci <hemi>.<label><letter>.label i.e. rh.MFSa.label   #####
#####                                                                       ##### 
#################################################################################    


for sub in subs:
    filepaths_dis = sorted(glob.glob(subject_dir+'/*/label/manual_label/*.label'))
    sub_dis = np.empty(shape=(len(filepaths_dis)),dtype=object)
    hemi_dis = np.empty(shape=(len(filepaths_dis)),dtype=object) 

    
for i, filepath in enumerate(filepaths_dis):
    sub_dis[i] = filepath.split('/')[-4]
    hemi_dis[i] = filepath.split('/')[-1].split('.')[0]
    
dis_pairs = list(zip(sub_dis,hemi_dis))

#################################################################################
#####            Calculate sulcal length and save to csv                    #####
#####                                                                       #####
#####  RESULT: Saves sulcal length (mm) to primary project folder as        #####
#####  <label_name>_path_length.csv                                         #####
#################################################################################                                   

##Create Dataframe for sulcal length
columns =  ['sub', 'hemi', 'label', 'max_path_length']
 
subject_id = []
hemi_id = []
df_anatomical = pd.DataFrame(columns = columns)

for label in labels:
    for sub_hemi in sub_hemi_combos: 
    
        sub = sub_hemi[0]
        hemi = sub_hemi[-1]
        label_stats = []
        label_name = label
        label_filename = subject_dir + '/{}/label/{}.{}.label'.format(sub, hemi, label)
        label_path_filename = subject_dir + '/{}/label/{}.{}.path'.format(sub, hemi, label)
        pair_tuple = tuple([sub,hemi])
        
        if pair_tuple in dis_pairs:
            manual_label_files = sorted(glob.glob(subject_dir +
                                                  '/{}/label/manual_label/{}.{}?.label'.format(sub, hemi,label)))
            path_lengths = np.empty(shape=(len(manual_label_files)),dtype=object)
            for i,path_length_file in enumerate(manual_label_files): 
                label_path_filename = subject_dir + '{}/label/manual_label/{}.{}?.path'.format(sub,hemi,str(i))                    
                path_lengths[i] = max_path_length(sub, hemi, path_length_file, label_path_filename)
            try:
                path_length=np.sum(path_lengths)
                label_stats = np.append(label_stats, path_length)
                subject_id = np.append(subject_id, sub)
                hemi_id = np.append(hemi_id, hemi)
            except TypeError:
                path_length = 'discontinuous'
        else:
            path_length = max_path_length(sub, hemi, label_filename, label_path_filename)
            label_stats = np.append(label_stats, path_length)
            subject_id = np.append(subject_id, sub)
            hemi_id = np.append(hemi_id, hemi)                  
                                            
        #Append values to label_stats and append all to dataframe  
        descriptives = [sub, hemi, label] 
        
        df_row = pd.DataFrame([descriptives + list(label_stats)], columns=columns)
        print('Finished with ', descriptives[0], descriptives[1], descriptives[2])
        df_anatomical = pd.concat([df_anatomical, df_row])

df_anatomical.to_csv('{}/../sulcal_length.csv'.format(subject_dir))

            