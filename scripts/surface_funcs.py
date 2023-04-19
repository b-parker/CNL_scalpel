# Utilities
from pathlib import Path
import os, sys
import subprocess as sp

# Data
import pandas as pd
import numpy as np

# Brain
import nibabel as nb
from nibabel.freesurfer.io import read_annot, read_label, read_morph_data, read_geometry
import cortex
import src.mesh_laplace_sulci

import gdist
import pygeodesic.geodesic as geodesic

# Plotting
from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

import matplotlib.pyplot as plt

from utility_funcs import mris_convert_command

# Meshes
import igl
import meshplot 






############################################################################################################
############################################################################################################
############################################################################################################


#                   Geodesic


############################################################################################################
############################################################################################################
############################################################################################################




# NOTE: had trouble with numba format and jit in surfdist, so some functions are copied over with slight modifications below


def translate_src(src, cortex):
    """
    Convert source nodes to new surface (without medial wall).
    """
    src_new = np.array(np.where(np.in1d(cortex, src))[0], dtype=np.int32)

    return src_new


def triangles_keep_cortex(triangles, cortex):
    """
    Remove triangles with nodes not contained in the cortex label array
    """

    # for or each face/triangle keep only those that only contain nodes within the list of cortex nodes
    input_shape = triangles.shape
    triangle_is_in_cortex = np.all(np.reshape(np.in1d(triangles.ravel(), cortex), input_shape), axis=1)

    cortex_triangles_old = np.array(triangles[triangle_is_in_cortex], dtype=np.int32)

    # reassign node index before outputting triangles
    new_index = np.digitize(cortex_triangles_old.ravel(), cortex, right=True)
    cortex_triangles = np.array(np.arange(len(cortex))[new_index].reshape(cortex_triangles_old.shape), dtype=np.int32)

    return cortex_triangles


def surf_keep_cortex(surf, cortex):
    # split surface into vertices and triangles
    vertices, triangles = surf

    # keep only the vertices within the cortex label
    cortex_vertices = np.array(vertices[cortex], dtype='float64')

    # keep only the triangles within the cortex label
    cortex_triangles = triangles_keep_cortex(triangles, cortex)

    return cortex_vertices, cortex_triangles


def dist_calc_matrix(surf, cortex, label_inds_all):
    cortex_vertices, cortex_triangles = surf_keep_cortex(surf, cortex)
    
    n_labels = len(labels)
 
    dist_mat = zeros((n_labels,n_labels))

    
    for r1 in arange(n_labels):
        #print('r1',r1,label_inds_all[r1])
        for r2 in arange(n_labels):
            # print('r2',r2,label_inds_all[r2])
            # val1 = gdist.compute_gdist(cortex_vertices, cortex_triangles,
            #                                source_indices = array(label_inds_all[r1]))
            # print('val1',val1)
            # Uncomment next three lines for Suvi original gdist code
            print('Label inds r1:', label_inds_all[r1])
            
            # val2 = gdist.compute_gdist(cortex_vertices, cortex_triangles,
            #                                 source_indices = array(label_inds_all[r1]),
            #                                 target_indices = array(label_inds_all[r2]))
            
            
            #
            #
            ## TODO in order for this to work we need to select single index for the source and target
            ## Options: take center, take nearest

            val2 = geodesic.PyGeodesicAlgorithmExact(cortex_vertices, cortex_triangles)
            source_indices = find_centroid(cortex_vertices[label_inds_all[r1]])
            target_indices = find_centroid(cortex_vertices[label_inds_all[r2]])

            ## .geodesicDistances(array) gets distance amoung all indices provided in array
            ## .geodesicDistance(source, target) requires using single source index and single target index

            val2_distance, val2_path = val2.geodesicDistance(source_indices, target_indices)
            # val2_distance, val2_path = val2.geodesicDistances(label_inds_all[r1])

            #print('val2',val2)
            dist_mat[r1,r2] = amin(val2) #UNCOMMENT for original code
            # dist_mat[r1,r2] = amin(val2_distance)

    return dist_mat

def find_centroid(label_vertices):
    """
    Finds centroid position for a list of RAS vertices

    returns list = [R, A, S]
    """ 
    centroid = []
    for i in arange(3):
        sum_verts = np.sum(label_vertices[i,:])
        centroid_val = sum_verts / len(label_vertices)
        centroid.append(centroid_val)
    return centroid

def getLabelIndices(sub,hemi,labels,cortex, subjects_dir):
    label_inds_all = []
    
    n_labels = len(labels)
    
    
    print('Num labels:', n_labels)

    for lab in labels:
        if type(lab) is list: # pick the first label in list that exists
            label_found = False
            for inner_label in lab:
                labelfile = '%s/%s/label/%s.%s.label'%(subjects_dir,sub,hemi,inner_label)
                if os.path.exists(labelfile) and not label_found:
                    labelfile_use = labelfile
                    label_found = True
        else: # look for specific label
            labelfile_use = '%s/%s/label/%s.%s.label'%(subjects_dir,sub,hemi,lab)
        label_inds = nib.freesurfer.io.read_label(labelfile_use, read_scalars=False)
        label_inds_t = translate_src(label_inds,cortex) # exclude medial wall
        label_inds_all.append(label_inds_t)
        
    
    return label_inds_all


def getDistMatrix(subjects_dir=str, labels=list, sub=str, hemi=str, savedir=str, fmri_prep=False):
    """
    Outputs geodesic distances among all labels for a given sub/hemi
    """
    if fmri_prep == True:
        highres_surface = '%s/sub-%s/ses-%s/anat/sub-%s_ses-%s_hemi-%s_midthickness.surf.gii'%(subjects_dir,sub,sub[-1],sub,sub[-1],hemi[0].upper())
    if fmri_prep == False:
        highres_surface = f'{subjects_dir}/{sub}/surf/{hemi}.pial'
    
 
    giidata = nb.freesurfer.read_geometry(highres_surface)

   
    # giidata2 = np.squeeze(np.asarray([x for x in giidata])) 
    surf = (giidata[0],giidata[1])  

    
    if fmri_prep == True:
        cort_file = '%s/sub-%s/label/%s.cortex.label'%(os.environ['SUBJECTS_DIR'],sub,hemi)
    if fmri_prep == False:
       cort_file = f'{subjects_dir}/{sub}/label/{hemi}.cortex.label'
    
    cortex = sort(nb.freesurfer.read_label(cort_file))

    
    label_inds_all = getLabelIndices(sub,hemi,labels,cortex, subjects_dir)

    dist_matrix = dist_calc_matrix(surf,cortex,label_inds_all)
   

    savetxt('%s/adj-labels-%s.txt'%(savedir,hemi),dist_matrix)


    



############################################################################################################
############################################################################################################
############################################################################################################


#                   Subject Class and Boundary functions


############################################################################################################
############################################################################################################
############################################################################################################


# Use boundary sulci to capture all vertices in surface between boundaries, and plot

import functools
from utility_funcs import memoize

class ScalpelSurface:
    """
    Surface class for performing scalpel operations

    REQUIREMENTS:
        - Freesurfer recon-all on subject
        - subject_filepath to subject freesurfer directory

    INPUT:
        - subject_filepath to subject freesurfer directory

    METHODS:
        cortex: list; [lh, rh], each is the result of nibabel.freesurfer.io.read_label on ?h.cortex.label i.e. [vertex_num, [R, A, S] for all vertices in hemisphere]
        make_roi_cut:  input - hemisphere, anterior, posterion, inferior, superior label names
                       output - freesurfer .label file including all vertices within bounded ROI

        get_boundary:  input - hemisphere=str, anterior=str, posterior=str, inferior=str, superior=str 
                       output - dict of boundary vertex numbers and vertex coordinates

        plot_boundary: input - label_name=str, label_filepath=str, outlier_corrected=bool, boundary_type=str, decimal_size=int 
                       output - 3D plots boundary vertices alongside vertices for whole label

    """
    def  __init__(self, subject_filepath= str):
        self._subject_filepath = Path(subject_filepath)
        

    @property
    @memoize
    def subject_filepath(self):
        return self._subject_filepath

    @subject_filepath.setter    
    def subject_filepath(self, value):
        print(f'"{self.subject_filepath}" is now "{value}"')
        self.subject_filepath = value

    @property
    @memoize
    def cortex(self):
        """
        Whole brain as a list with two elements, [lh, rh]
        each hemi is an array with two elements, [vertex_index, RAS_coords]
        """
        cortex = [read_label(self._subject_filepath / 'label/lh.cortex.label'), # lh.cortex.label
                  read_label(self._subject_filepath / 'label/rh.cortex.label')] # rh.cortex.label
        
        return cortex
    
    @cortex.setter
    def cortex(self, value):
        __subject_filepath = Path(value)
        cortex = [read_label(self._subject_filepath / 'label/lh.cortex.label'), # lh.cortex.label
                  read_label(self._subject_filepath / 'label/rh.cortex.label')] # rh.cortex.label
        return cortex
    
    


    def get_surface(self, surface_type, hemi):
        """Reads morph data on hemi.surface_type"""
        return read_morph_data(self.subject_filepath / f'surf/{hemi}.{surface_type}')
        
    
    def get_boundary(self, hemi, anterior_label, posterior_label, inferior_label, superior_label):
        """ 
        Gets edge of labels along anterior, posterior, inferior, superior labels

        Returns dictionary of {direction : [RAS_coordinates, vertex_number] 
        """
        anterior_boundary_vertices, anterior_boundary_vertex_num = find_boundary_vertices(subject_filepath=self._subject_filepath, hemi=hemi, boundary_type='anterior', label_name=anterior_label, outlier_corrected=True, decimal_size=1)
        posterior_boundary_vertices, posterior_boundary_vertex_num = find_boundary_vertices(subject_filepath=self._subject_filepath, hemi=hemi, boundary_type='posterior', label_name=posterior_label, outlier_corrected=True, decimal_size=1)
        inferior_boundary_vertices, inferior_boundary_vertex_num = find_boundary_vertices(subject_filepath=self._subject_filepath, hemi=hemi, boundary_type='inferior', label_name=inferior_label, outlier_corrected=True, decimal_size=1)
        superior_boundary_vertices, superior_boundary_vertex_num = find_boundary_vertices(subject_filepath=self._subject_filepath, hemi=hemi, boundary_type='superior', label_name=superior_label, outlier_corrected=True, decimal_size=1)

        boundary_dict = {'anterior' : [anterior_boundary_vertices, anterior_boundary_vertex_num],
                         'posterior' : [posterior_boundary_vertices, posterior_boundary_vertex_num],
                         'inferior' : [inferior_boundary_vertices, inferior_boundary_vertex_num ],
                         'superior' : [superior_boundary_vertices, superior_boundary_vertex_num]}
        # edge_points = {'anterior_lh' : [np.min, np.max], 'anterior_rh': [np.min, np.max], 
        #                'posterior_lh' :  } ## TODO finish iding edge points and then get geodesic paths among them
        
        # ID hemi
        if hemi == 'lh':
            hemi_ind = 0
        else:
            hemi_ind = 1

        # geodesic line drawn between vertices TODO
        # geoalg = geodesic.PyGeodesicAlgorithmExact(self.cortex[hemi_ind])


        return boundary_dict
    
    def make_ROI_cut(self, anterior_label, posterior_label, inferior_label, superior_label, ROI_name='', hemi=''):
        # Makes label of entire ROI between boundary vertices
        boundary_verts = self.get_boundary(hemi=hemi, anterior_label=anterior_label, posterior_label=posterior_label, inferior_label=inferior_label, superior_label=superior_label)
        
        for i, key in enumerate(boundary_verts.keys()):
            direction_sets = boundary_verts[key]
            


        # Gets rounded bins for each axis
        rounded_anterior = np.round(boundary_verts['anterior'][1][:, 1], decimals=1)
        rounded_posterior = np.round(boundary_verts['posterior'][1][:, 1], decimals=1)
        rounded_inferior = np.round(boundary_verts['inferior'][1][:, 2], decimals=1)
        rounded_superior = np.round(boundary_verts['superior'][1][:, 2], decimals=1)
        print(boundary_verts)

        cut_roi_coords  = []
        cut_roi_vert_idx = []
        # get vertex indices for all within ROI
        for hemi in self.cortex:
            hemi_cortex_idx = hemi[0]
            hemi_cortex_coords = hemi[1]
            for point in hemi_cortex_coords:
                point_rounded_ant_post = np.round(point[1], decimals=1)
                point_rounded_inf_sup = np.round(point[2], decimals=1)
                # find nearest point in rounded array
                

                # add all points within boundaries to cut_roi lists

                # draw geodesic path
                



        

    def plot_boundary(self, label_name='label', label_filepath='', outlier_corrected_bool=True, boundary_type='anterior', decimal_size=1):
        # Plot boundary versus original label
        vertices, coords = read_label(label_filepath)
        
        r_data = np.array([ras[0] for ras in coords])
        a_data = np.array([ras[1] for ras in coords])
        s_data = np.array([ras[2] for ras in coords])

        boundary_vert_num, boundary_verts = find_boundary_vertices(boundary=boundary_type, label_name=[vertices, coords], outlier_corrected=outlier_corrected_bool, decimal_size=decimal_size)

        boundary_r_data = np.array([ras[0] for ras in boundary_verts])
        boundary_a_data = np.array([ras[1] for ras in boundary_verts])
        boundary_s_data = np.array([ras[2] for ras in boundary_verts])

        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax1.scatter3D(r_data, a_data, s_data, cmap='viridis', c=s_data)

        # Getting lower bound of the sulcus

        ax2.scatter3D(boundary_r_data, boundary_a_data, boundary_s_data, cmap='viridis', c=boundary_s_data, vmin=10, vmax=45)
        ax2.set(zlim3d=[10,45]);
        plt.suptitle(f'all vertices of {label_name} vs {boundary_type} boundary')
        plt.show;




def find_boundary_vertices(subject_filepath, hemi, label_name, boundary_type, outlier_corrected=True, decimal_size=1, outlier_std_threshold=3):
    """
    Finds vertices along boundary of given Freesurfer label, as identified by boundary_type.

    Bins all vertices along boundary axis, effectively getting every vertex in a slice rounded to the decimal_size,
    and returns coordinates and vertex numbers with the minimum or maximum axis value as specified by boundary type.

    INPUT:
    label_name = string : as it appears in /label/ freesurfer directory ie MCGS for lh.MCGS.label
    outlier_corrected = boolean; if True removes all vertices if S value is outside of 3 SDs of mean
    decimal_size = int; determines size of rounding done to bin superior vertices, 1 is suggested
    outlier_std_threshold= int; allows change of SD multiple for outlier correction

    OUTPUT:
    boundary_vertex_num = np array; vertex number for each minimum Superior value along superior axis
    boundary_coords = np array; RAS coordinates for each minimum Superior value along superior axis


    Example:

    output_vertex_number, output_vertex_coords = find_boundary_vertices(subject_filepath='~/subjects/100206/', hemi='lh', label_name='MCGS', boundary_type='anterior')


    """
    if boundary_type == 'anterior' or boundary_type == 'posterior':
      # orthogonal measure index is the index along which we are trying to find min values i.e. A for S
      # finding minimum values on boundary_type axis along orthogonal axis
      measure_idx = 1
      orthogonal_measure_idx = 2

    if boundary_type == 'inferior' or boundary_type == 'superior':
      measure_idx = 2
      orthogonal_measure_idx = 1
    
    label_data = read_label(subject_filepath / f"label/{hemi}.{label_name}.label")

    vertex_num = label_data[0]
    ras_coords = label_data[1]

    r_data = np.array([ras[0] for ras in ras_coords])
    a_data = np.array([ras[1] for ras in ras_coords])
    s_data = np.array([ras[2] for ras in ras_coords])
    
    all_data = [r_data, a_data, s_data]
    measure_data = all_data[measure_idx]
    orthogonal_data = all_data[orthogonal_measure_idx]
    
    
    boundary_coords = []
    boundary_vertex_num = []
    rounded_orthogonal = np.round(orthogonal_data, decimals=decimal_size)

  # for each rounded A value, find the vertex with the lowest S value
  # add vertex to posterior boundary

    for orthogonal_edge in np.unique(rounded_orthogonal):
      # get all vertices with shared orthogonal coordinate
      column_idx = np.where(rounded_orthogonal == orthogonal_edge)[0]

      # find minimum coordinate from that column of points
      if boundary_type == 'anterior' or boundary_type == 'inferior':
        min_val = np.amin(measure_data[column_idx])
        # be sure idx is drawn from original column idxes
        all_min_idx = np.where(measure_data == min_val)[0]
        column_min_idx = np.intersect1d(all_min_idx, column_idx)
        # add boundary and vertex to list
        boundary_coords.append(list(ras_coords[column_min_idx][0]))
        boundary_vertex_num.append(vertex_num[column_min_idx][0])
        
      else:
        max_val = np.amax(measure_data[column_idx])
        # be sure idx is drawn from original column idxes
        all_max_idx = np.where(measure_data == max_val)[0]
        column_max_idx = np.intersect1d(all_max_idx, column_idx)
        # add posterior boundary and vertex to list
        boundary_coords.append(list(ras_coords[column_max_idx][0]))
        boundary_vertex_num.append(vertex_num[column_max_idx][0])
      

    mean_boundary_coord = np.mean([i[measure_idx] for i in boundary_coords])
    std_boundary_coord = np.std([i[measure_idx] for i in boundary_coords])
    
    # Keep coordinate if coordinate is within 3 stds of mean 
    if outlier_corrected == True:
      boundary_coords_outlier = []
      boundary_vertex_num_outlier = []
      for i, coord_vert in enumerate(boundary_coords):
          if coord_vert[measure_idx] > (mean_boundary_coord - decimal_size * std_boundary_coord) and coord_vert[measure_idx] < (mean_boundary_coord + decimal_size * std_boundary_coord): 
            boundary_coords_outlier.append(boundary_coords[i])
            boundary_vertex_num_outlier.append(boundary_vertex_num[i])
          else:
            pass
      return np.array(boundary_vertex_num_outlier), np.array(boundary_coords_outlier)
    else: 
      pass
#     for vertex in ras_coords:
      return np.array(boundary_vertex_num), np.array(boundary_coords)
    


def get_faces_from_vertices(faces : np.array, label_ind : np.array):
    """
    Takes a list of faces and label indices
    Returns the faces that contain the indices

    INPUT:
    faces: array of faces composed of 3 points
    label_ind: array of indices of points in the label (first colum of label file; 0 index in read_label)

    OUTPUT:
    label_faces: array of faces that contain the points in the label
    """
    all_label_faces = []
    for face in faces:
        for point_index in face:
            if point_index in label_ind:
                all_label_faces.append(list(face))
    return np.array(all_label_faces)

def get_boundary_faces(all_faces : np.array, label_ind : np.array):
    """
    For a given label, find the faces that are on the boundary of the label by finding vertices
    that only appear twice in the array of faces (all interior vertices will appear 3 or more times)

    INPUT:
        all_faces : np.array - array of faces from the mesh
        label_ind : np.array - array of vertices that are in the label, first column in freesurfer .label file
    OUTPUT:
        boundary_faces : np.array - array of faces that are on the boundary of the label
    """
    # Find the unique faces of a label
    faces_in_label = get_faces_from_vertices(faces, label_ind)
    unique_entry, count = np.unique(faces_in_label, return_counts=True)
    # Get the nodes that only appear once
    boundary_nodes = unique_entry[count <= 2]
    # Get the faces that include the boundary nodes
    boundary_faces = get_faces_from_vertices(faces, boundary_nodes)
    return boundary_faces



############################################################################################################
############################################################################################################
############################################################################################################


                #   Graph Functions


############################################################################################################
############################################################################################################
############################################################################################################

def mesh_to_adjacency(all_faces, all_points):
    """
    Turn a triangular mesh into an adjacency matrix for traversal 
    """
    adjacency = np.zeros((len(all_points), len(all_points))) # Same indexes as point-vertexes

    for i in np.arange(len(all_faces)):
        face = all_faces[i]
        v1, v2, v3 = face
        adjacency[v1,v2] = 1
        adjacency[v2,v1] = 1
        adjacency[v1,v3] = 1
        adjacency[v3,v1] = 1
        adjacency[v2,v3] = 1
        adjacency[v3,v2] = 1

    return adjacency

def adjacenct_nodes(adjacency_matrix : np.array, vertex : int):
    """
    Find adjacent nodes to a given vertex

    INPUT:
    adjacency_matrix: np.array - adjacency matrix of a mesh provided by mesh_to_adjacency()
    vertex: int - index of vertex to find adjacent nodes 

    OUTPUT:
    adjacenct_vertices: list - list of indexes of adjacent vertices

    """
    vertex_adjacency = adjacency_matrix[vertex]
    adjacenct_vertices = [idx for idx, val in enumerate(vertex_adjacency) if val == 1]
    return adjacenct_vertices


def get_vertices_in_bounded_area(all_faces, all_points, boundary_faces):
    """
    For a list of boundary faces, start at the most posterior node (node 1)

    find all adjacent faces, and select the faces with the most anterior node (node 2)

    Using the faces defined by node 1 and node 2, breadth first search until you encounter boundary faces and there are no more unvisited nodes

    INPUT:
    all_faces: np.array - array of faces in mesh
    all_points: np.array - array of points in mesh
    boundary_faces: np.array - array of boundary faces


    """

    ## add points in boundary faces to label_points 
    label_points = boundary_faces.flatten().unique()
    
    # find most posterior point in boundary
    all_boundary_points = all_points[boundary_faces]
    post_boundary_point_value = [points[1] for points in all_boundary_points].min()
    post_boundary_point = all_points[np.where(all_points[1] == post_boundary_point_value)]

    # get adjacent nodes to boundary
    post_adjacent_nodes = adjacenct_nodes(post_boundary_point, all_faces)

    # get the most anterior point in all nodes adjacent to the most posterior point
    first_anterior_point = [point for point in np.take(points, post_adjacent_nodes, axis=0)[1]].flatten().min()

    first_anterior_node = all_points[np.where(all_points[1] == first_anterior_point)]

    # breadth first search from first anterior point, treating boundary points as end of the graph
    queue = [first_anterior_node]

    while queue:
        vertex = queue.pop(0)
        if vertex not in label_points:
            label_points.append(vertex)
            for adj in adjacenct_nodes(vertex):
                if adj not in label_points:
                    queue.append(adj)

