# Utilities
from pathlib import Path
import os, sys
import subprocess as sp
from functools import partial 

# Data
import pandas as pd
import numpy as np

# Brain
import nibabel as nb
from nibabel.freesurfer.io import read_annot, read_label, read_morph_data, read_geometry
import cortex
import src.mesh_laplace_sulci

import gdist
import surfdist
import pygeodesic.geodesic as geodesic

# Plotting
from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

import matplotlib.pyplot as plt

from src.utility_funcs import mris_convert_command

# Meshes
import igl
import networkx as nx
#import meshplot 
from src.freesurfer_utils import *






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
from src.utility_funcs import memoize

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

def find_label_boundary_vertices(label_faces):
    """
    Find the boundary edges of a label

    INPUT:
    label_faces: np.array - array of faces in a label

    OUTPUT:
    boundary_edges: np.array - array of boundary edges in a label
    """
    from collections import Counter
    edges = Counter()
    for face in label_faces:
    
        edges.update([tuple(sorted([face[i], face[j]])) for i in range(3) for j in range(i + 1, 3)])
    boundary_edges = [edge for edge, count in edges.items() if count == 1]

    return np.unique(boundary_edges)



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

def adjacent_nodes(adjacency_matrix : np.array, vertex : int):
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

def find_label_boundary(label_faces):
    """
    Find the boundary edges of a label

    INPUT:
    label_faces: np.array - array of faces in a label

    OUTPUT:
    boundary_edges: np.array - array of boundary edges in a label
    """
    from collections import Counter
    edges = Counter()
    for face in label_faces:
    
        edges.update([tuple(sorted([face[i], face[j]])) for i in range(3) for j in range(i + 1, 3)])

    
    boundary_edges = [edge for edge, count in edges.items() if count == 1]

    return np.unique(boundary_edges)

def find_endpoint_vertices(path: list, graph: nx.Graph):
    """
    Find the vertices within a connected graph which only share a single connections to the rest of the graph. 
    These vertices are the endpoints of the graph

    INPUT:
    path: list - list of vertices in the path
    graph: nx.Graph - graph of the mesh

    OUTPUT:
    endpoints: list - list of endpoints in the graph

    """
    path_graph = graph.subgraph(path).copy()
    vertices = list(path_graph.nodes)
    edge = []
    for vertex in vertices:
        adj_nodes = list(path_graph.adj[vertex])
        num_connections = len([v for v in adj_nodes if v in vertices])
        if num_connections == 1:
            edge.append(vertex)

    return edge



def find_vert_inside(adjacency_matrix: np.array, vert : int, all_verts : np.array, label_verts : np.array, direction : str = 'anterior'):
        """ 
        Finds the first vertex on a graph inside of a boundary, according to a direction. 
        i.e. 'anterior' will find the first anterior vertex inside the <vert> given

        INPUT:
        adjacency_matrix: np.array - adjacency matrix of a mesh provided by mesh_to_adjacency()
        vert: int - index of vertex to find adjacent nodes
        all_verts: np.array - array of all vertices in mesh
        label_verts: np.array - array of vertices in boundary
        direction: str - direction to search for first point inside boundary

        OUTPUT:
        first_vert_index: int - index of first vert inside boundary
        """

        ## Recursively check for first point anterior to boundary
        adjacent_points = adjacent_nodes(adjacency_matrix, vert)

        if direction == 'anterior':
            dir_idx = 1
            dir_function = np.min
        if direction == 'posterior':
            dir_idx = 1
            dir_function = np.max
        if direction == 'inferior':
            dir_idx = 2
            dir_function = np.min
        if direction == 'superior':
            dir_idx = 2
            dir_function = np.max                  
        all_verts_in_direction = np.array([vert_i for vert_i in np.take(all_verts, adjacent_points, axis=0)[dir_idx]]).flatten()
        first_point = dir_function(all_verts_in_direction)
        if first_point in label_verts:
            find_vert_inside(first_point)
        else:
            ## Base 
            ### Return index and value
            first_vert_index = np.where(all_verts[:, dir_idx] == first_point)[0][0] 
            return first_vert_index 
        
  
  
def find_edge_vert(label_RAS: np.array, label_ind: np.array, direction: str, hemi: str):
    """
    Finds the vertex on the boundary edge in a given direction

    INPUT:
    label_verts: np.array - array of vertices in boundary
    direction: str - direction to search for first point inside boundary

    OUTPUT:
    first_vert_index: int - index of first vert inside boundary
    """
    if direction == 'anterior':
        dir_idx = 1
        dir_function = np.max
    elif direction == 'posterior':
        dir_idx = 1
        dir_function = np.min
    elif direction == 'inferior':
        dir_idx = 2
        dir_function = np.min
    elif direction == 'superior':
        dir_idx = 2
        dir_function = np.max
    elif direction == 'medial' and hemi == 'lh':
        dir_idx = 0
        dir_function = np.max
    elif direction == 'lateral' and hemi == 'lh':
        dir_idx = 0
        dir_function = np.min
    elif direction == 'medial' and hemi == 'rh':
        dir_idx = 0
        dir_function = np.min
    elif direction == 'later' and hemi == 'rh':
        dir_idx = 0
        dir_function = np.max

    first_point = dir_function(label_RAS[:, dir_idx])
    first_RAS = label_RAS[np.where(label_RAS[:, dir_idx] == first_point)]
    first_vert_index = label_ind[np.where(label_RAS[:, dir_idx] == first_point)]

    return np.array(first_vert_index), np.array(first_RAS)


def get_vertices_in_bounded_area(all_faces, all_points, boundary_faces):
    """
    For a list of boundary faces, start at the most posterior node (node 1)

    find all adjacent faces, and select the faces with the most anterior node (node 2)

    Using the faces defined by node 1 and node 2, breadth first search until you encounter boundary faces and there are no more unvisited nodes

    INPUT:
    all_faces: np.array - array of faces in mesh
    all_points: np.array - array of points in mesh
    boundary_verts: np.array - array of boundary vertex indices

    OUTPUT:
    label_points: np.array - array of points in bounded area

    """
    # Get all points in boundary
    label_points = np.unique(boundary_faces)

    # Calculate adjacency matrix for traversal
    adj_mat = mesh_to_adjacency(all_faces, all_points)
    
    # return the index of the most posterior point in the boundary (idx in all_points)
    all_boundary_points = all_points[label_points]
    boundary_index = find_edge_vert(all_boundary_points, 'posterior')
            
    first_anterior_vertex = find_vert_inside(adjacency_matrix=adj_mat,
                                            vert=boundary_index,
                                            all_verts=all_points,
                                            label_verts=label_points,
                                            direction='anterior')

    # breadth first search from first anterior point, treating boundary points as end of the graph
    queue = [first_anterior_vertex]

    while queue:
        visited = label_points
        vertex = queue.pop(0)
        if vertex not in label_points:
            ## Seems infinite
            ## Not adding any points
            np.append(label_points, vertex)
            for adj in adjacenct_nodes(adj_mat, vertex):
                if adj not in visited:
                    queue.append(adj)
                    np.append(visited, adj)
    
    return label_points


def get_label_subsets(label_faces: np.array, all_faces: np.array) -> list:
    """
    Get the disjoint sets of a label

    INPUT:
    label_faces: np.array - array of faces in a label
    all_faces: np.array - array of all faces in a mesh

    OUTPUT:
    dj_set: list - list of disjoint sets of the label
    """

    from scipy.cluster.hierarchy import DisjointSet

    dj_set = DisjointSet(np.unique(label_faces))

    for triangular_face in label_faces:
        dj_set.merge(triangular_face[0], triangular_face[1])
        dj_set.merge(triangular_face[0], triangular_face[2])

    dj_set = [get_faces_from_vertices(all_faces, subset) for subset in dj_set.subsets()]
    return dj_set

def create_graph_from_mesh(faces):
    """
    Create a graph from a mesh

    INPUT:
    faces: np.array - array of faces in mesh

    OUTPUT:
    G: nx.Graph - graph of the mesh
    """

    G = nx.Graph()

    # Add edges based on faces
    for face in faces:
        for i in range(3):
            for j in range(i+1, 3):
                # Add an edge between the vertices of each triangle
                G.add_edge(face[i], face[j])

    return G

def find_shortest_path_in_mesh(faces, source_index, target_index):
    """
    Find the shortest path between two vertices in a mesh

    INPUT:
    faces: np.array - array of faces in mesh
    source_index: int - index of source vertex
    target_index: int - index of target vertex

    OUTPUT:
    path: list - list of vertices in the shortest path

    """
    
    # Create a graph from the mesh
    G = create_graph_from_mesh(faces)

    # Find the shortest path
    path = nx.shortest_path(G, source=source_index, target=target_index)

    return path


def make_roi_cut(anterior: str, posterior: str, superior: str, inferior: str, hemi: str, all_points: np.array, all_faces: np.array, subjects_dir: str or Path, sub: str):
    """ 
    Create an outlined region of cortex, using each of the labels as the boundary in a given direction. Take 4 labels names, hemisphere, and mesh, and create a single label that is the bounded roi.

    INPUT: 
    anterior: str - name of anterior label
    posterior: str - name of posterior label
    superior: str - name of superior label
    inferior: str - name of inferior label
    hemi: str - hemisphere of interest
    all_points: np.array - array of all points in a hemisphere as loaded by nibabel freesurfer.io.read_geometry
    all_faces: np.array - array of all faces in a hemisphere as loaded by nibabel freesurfer.io.read_geometry
    subjects_dir: str or Path - path to freesurfer subjects directory
    sub: str - subject id

    OUTPUT:
    roi_label_ind: np.array - array of indices of vertices in the bounded roi
    roi_label_points: np.array - array of points in the bounded roi

    TODO
    - fractionated sulci
    - angle of direction
    - different path length algorithms
    - maybe using only boundary vertices to create a larger ROI for a given portion of the brain
    - splitting a region 

    
    """


    labels = {'anterior': [anterior], 'posterior': [posterior], 'superior': [superior], 'inferior': [inferior]}

    inflated_surface = nb.freesurfer.io.read_geometry(f'{subjects_dir}/{sub}/surf/{hemi}.inflated')

    for i in labels.keys():
        raw_label = read_label(f'{subjects_dir}/{sub}/label/{hemi}.{labels[i][0]}.label')
        
        inflated_RAS = np.array(inflated_surface[0][raw_label[0]])

        labels[i].append((raw_label[0], inflated_RAS))

    edge_points = {}

    ## Get edge points for each boundary label
    for boundary in labels.keys():
        if boundary == 'anterior' or boundary == 'posterior':
            edges = ['superior', 'inferior']
            label_ind = labels[boundary][1][0]
            label_RAS = labels[boundary][1][1]
            label_name = labels[boundary][0]
            edge_points[f'{boundary}_{label_name}_label_{edges[0]}_edge'] = find_edge_vert(label_RAS, label_ind, edges[0])
            edge_points[f'{boundary}_{label_name}_label_{edges[1]}_edge'] = find_edge_vert(label_RAS, label_ind, edges[1])

        else:
            edges = ['anterior', 'posterior']
            label_ind = labels[boundary][1][0]
            label_RAS = labels[boundary][1][1]
            label_name = labels[boundary][0]
            edge_points[f'{boundary}_{label_name}_label_{edges[0]}_edge'] = find_edge_vert(label_RAS, label_ind, edges[0])
            edge_points[f'{boundary}_{label_name}_label_{edges[1]}_edge'] = find_edge_vert(label_RAS, label_ind, edges[1])

    ## Get the shortest path between the paired points on the boundary
    boundary_paths = {}

    for ant_post in ["anterior", "posterior"]:
        for sup_inf in ["superior", "inferior"]:
            starting_vertex = edge_points[f"{ant_post}_{labels[ant_post][0]}_label_{sup_inf}_edge"][0][0]
            target_vertex = edge_points[f"{sup_inf}_{labels[sup_inf][0]}_label_{ant_post}_edge"][0][0]

            path = find_shortest_path_in_mesh(all_points, all_faces, starting_vertex, target_vertex)
            boundary_faces = get_faces_from_vertices(all_faces, path)
            boundary_paths[f"{ant_post}_{sup_inf}_{labels[ant_post][0]}_to_{sup_inf}_{ant_post}_{labels[sup_inf][0]}"] = path
            #boundary_paths[f"{ant_post}_{sup_inf}_{labels[ant_post][0]}_to_{sup_inf}_{ant_post}_{labels[sup_inf][0]}"] = boundary_faces

    ## Combine all labels and paths into a single label
    
    roi_label_ind = np.unique(np.concatenate([labels[i][1][0] for i in labels.keys()]))
    print(len(roi_label_ind))
    for i in boundary_paths.keys():
        print(boundary_paths[i])
        roi_label_ind = np.unique(np.concatenate([roi_label_ind, boundary_paths[i]], axis=None))
        print(len(roi_label_ind))
    roi_label_points = all_points[roi_label_ind]
    return [roi_label_ind, roi_label_points]

from src.surface_funcs import get_label_subsets

def sort_sets_by_position(label_sets, points, direction):
    """
    Sort a list of faces by their average RAS position, according to direction. If direction is "anterior", 
    sort by the second element of the RAS point, if "superior", sort by the third element of the RAS point

    INPUT:
    label_sets: list - list of faces in label
    points: np.array - array of points in mesh
    direction: str - direction to sort by

    OUTPUT:
    sorted_sets: list - list of sorted faces
   """
    if direction == 'anterior':
        dir_idx = 1
    elif direction == 'superior':
        dir_idx = 2
    else:
        raise ValueError('direction must be anterior or superior')
    sorted_sets = sorted(label_sets, key=lambda x: np.median(points[x][:,dir_idx]))
    return sorted_sets

def make_2cut_RAS(label_1_RAS, label_1_ind, label_2_RAS, label_2_ind, direction: str, hemi: str, points, faces, subjects_dir, sub):
    """  
    Create ROI between two sulci by connecting edges of the sulci according to a direction - anterior-posterior or superior-inferior

    INPUT:
    label_1: str - name of first label
    label_2: str - name of second label
    direction: str - direction to connect edges "anterior-posterior" or "superior-inferior"
    hemi: str - hemisphere
    points: np.array - array of points in mesh
    faces: np.array - array of faces in mesh
    subjects_dir: str - path to subjects directory
    sub: str - subject ID

    OUTPUT:
    roi_faces: np.array - array of faces in ROI
    roi_points: np.array - array of points in ROI
    """


    inflated_surface = nb.freesurfer.read_geometry(f'{subjects_dir}/{sub}/surf/{hemi}.inflated')
    label_1_faces = get_faces_from_vertices(faces, label_1_ind)
    label_2_faces = get_faces_from_vertices(faces, label_2_ind)

    ### get disjoint sets for each label
    label_1_subsets = get_label_subsets(label_1_faces)
    label_2_subsets = get_label_subsets(label_2_faces)

    ## sort subsets by their anterior posterior value from RAS
    ## return most posterior subset of label 1 and most anterior subset of label 
    label_1_subsets = sort_sets_by_position(label_1_subsets, points, direction)[-1]
    label_2_subsets = sort_sets_by_position(label_2_subsets, points, direction)[0]


    if direction == 'anterior-posterior':
        label_1_medial = find_edge_vert(label_1_RAS, label_1_ind, 'medial', hemi)[0][0]
        label_2_medial = find_edge_vert(label_2_RAS, label_2_ind, 'medial', hemi)[0][0]
        label_1_lateral = find_edge_vert(label_1_RAS, label_1_ind, 'lateral', hemi)[0][0]
        label_2_lateral = find_edge_vert(label_2_RAS, label_2_ind, 'lateral', hemi)[0][0]

    elif direction == 'superior-inferior':
        label_1_anterior = find_edge_vert(label_1_RAS, label_1_ind, 'anterior', hemi)[0][0]
        label_2_anterior = find_edge_vert(label_2_RAS, label_2_ind, 'anterior', hemi)[0][0]
        label_1_posterior = find_edge_vert(label_1_RAS, label_1_ind, 'posterior', hemi)[0][0]
        label_2_posterior = find_edge_vert(label_2_RAS, label_2_ind, 'posterior', hemi)[0][0]
    
    else:
        raise ValueError('direction must be anterior-posterior or superior-inferior')

    path1 = find_shortest_path_in_mesh(points, faces, label_1_medial, label_2_medial)
    path2 = find_shortest_path_in_mesh(points, faces, label_1_lateral, label_2_lateral)

    roi_paths = np.unique(np.concatenate((path1, path2)))
    roi_ind = np.concatenate((label_1_ind, label_2_ind, roi_paths))
    roi_RAS = np.concatenate((label_1_RAS, label_2_RAS, points[roi_paths]))

    return roi_ind, roi_RAS

def find_closest_vertices(boundary1: np.array, boundary2: np.array, points, faces, num_vertices: float = .1, path_length: bool = True):
    """
    Find the num vertices closest vertices for boundary1 and boundary2

    INPUT:
    boundary1: np.array - array of boundary vertices
    boundary2: np.array - array of boundary vertices
    points: np.array - array of points in mesh
    faces: np.array - array of faces in mesh
    num_vertices: float - percentage of each boundary to return
    path_length: bool - if True, use path length instead of euclidean distance
    
    OUTPUT:
    closest_vertices: np.array - array of closest vertices
    """
    if path_length:
        boundary1_closest = []
        boundary2_closest = []
        boundary1 = boundary1[::20]
        boundary2 = boundary2[::10]
        for i in range(len(boundary1)):
            for j in range(len(boundary2)):
                print(str(i) + f"/ {str(len(boundary1))}"), print( str(j)+ f"/ {str(len(boundary2))}")
                path = find_shortest_path_in_mesh(points, faces, boundary1[i], boundary2[j])
                boundary1_closest.append(path[0])
                boundary2_closest.append(path[-1])
        ## sort boundary1_closest and boundary2_closest by the path length
        boundary1_closest = np.array(boundary1_closest)
        boundary2_closest = np.array(boundary2_closest)
        boundary1_closest = boundary1_closest[np.argsort(boundary1_closest[:,1])]
        boundary2_closest = boundary2_closest[np.argsort(boundary2_closest[:,1])]

        ## return the len(boundary1) * num_vertices closest vertices and len(boundary2) * num_vertices closest vertices
        boundary1_closest = boundary1_closest[:int(len(boundary1) * num_vertices)]
        boundary2_closest = boundary2_closest[:int(len(boundary2) * num_vertices)]
        return boundary1_closest, boundary2_closest
    
    else:
        boundary1_closest = []
        boundary2_closest = []
        for i in range(len(boundary1)):
            for j in range(len(boundary2)):
                dist = np.linalg.norm(points[boundary1[i]] - points[boundary2[j]])
                boundary1_closest.append(dist)
                boundary2_closest.append(dist)
        ## sort boundary1_closest and boundary2_closest by the path length
        boundary1_closest = np.array(boundary1_closest)
        boundary2_closest = np.array(boundary2_closest)
        boundary1_closest = boundary1_closest[np.argsort(boundary1_closest[:,1])]
        boundary2_closest = boundary2_closest[np.argsort(boundary2_closest[:,1])]

        ## return the len(boundary1) * num_vertices closest vertices and len(boundary2) * num_vertices closest vertices
        boundary1_closest = boundary1_closest[:int(len(boundary1) * num_vertices)]
        boundary2_closest = boundary2_closest[:int(len(boundary2) * num_vertices)]
        return boundary1_closest, boundary2_closest
    




############################################################################################################
############################################################################################################
############################################################################################################


                #   Cluster-based parcellation


############################################################################################################
############################################################################################################
############################################################################################################


from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift

def cluster_label_OPTICS(label_ind, label_RAS, points, faces, min_samples: int = 5, eps: float = 1.5):
    """
    Cluster a label using OPTICS clustering

    INPUT:
    label_ind: np.array - array of indices of label
    label_RAS: np.array - array of RAS coordinates of label
    points: np.array - array of points in mesh
    faces: np.array - array of faces in mesh
    min_samples: int - minimum number of samples in a cluster
    eps: float - maximum distance between two samples for one to be considered in the same cluster

    OUTPUT:
    clusters: np.array - array of clusters
    """
    label_points = points[label_ind]
    clustering = OPTICS(min_samples=min_samples, eps=eps).fit(label_points)
    clusters = clustering.labels_
    return clusters

def cluster_label_DBSCAN(label_ind, label_RAS, points, faces, eps: float = 1.5):
    """
    Cluster a label using DBSCAN clustering

    INPUT:
    label_ind: np.array - array of indices of label
    label_RAS: np.array - array of RAS coordinates of label
    points: np.array - array of points in mesh
    faces: np.array - array of faces in mesh
    eps: float - maximum distance between two samples for one to be considered in the same cluster

    OUTPUT:
    clusters: np.array - array of clusters
    """
    label_points = points[label_ind]
    clustering = DBSCAN(eps=eps).fit(label_points)
    clusters = clustering.labels_
    return clusters

def cluster_label_KMeans(label_ind, label_RAS, points, faces, n_clusters: int = 2):
    """
    Cluster a label using KMeans clustering

    INPUT:
    label_ind: np.array - array of indices of label
    label_RAS: np.array - array of RAS coordinates of label
    points: np.array - array of points in mesh
    faces: np.array - array of faces in mesh
    n_clusters: int - number of clusters

    OUTPUT:
    clusters: np.array - array of clusters
    """
    label_points = points[label_ind]
    clustering = KMeans(n_clusters=n_clusters, n_init='auto').fit(label_points)
    clusters = clustering.labels_
    return clusters

def cluster_label_mean_shift(label_ind, label_RAS, points, faces, bandwidth: float = 1.5):
    """
    Cluster a label using mean shift clustering

    INPUT:
    label_ind: np.array - array of indices of label
    label_RAS: np.array - array of RAS coordinates of label
    points: np.array - array of points in mesh
    faces: np.array - array of faces in mesh
    bandwidth: float - bandwidth to use for mean shift clustering

    OUTPUT:
    clusters: np.array - array of clusters
    """
    label_points = points[label_ind]
    clustering = MeanShift(bandwidth=bandwidth).fit(label_points)
    clusters = clustering.labels_
    return clusters

## in cluster_kmeans, separate the clusters based on whether they are in label_1 or label_2

def separate_clusters(cluster_labels, label_1_ind, label_2_ind, combined_labels):
    """
    Separate clusters based on whether they are in label_1 or label_2

    INPUT:
    cluster_labels: np.array - array of cluster labels
    label_1_ind: np.array - array of indices of label_1
    label_2_ind: np.array - array of indices of label_2
    combined_labels: np.array - array of indices of label_1 and label_2

    OUTPUT:
    label_1_clusters: np.array - array of cluster labels for label_1
    label_2_clusters: np.array - array of cluster labels for label_2
    """
    label_1_clusters = cluster_labels[np.isin(combined_labels, label_1_ind)]
    label_2_clusters = cluster_labels[np.isin(combined_labels, label_2_ind)]
    return label_1_clusters, label_2_clusters


def find_closest_clusters(label_1_RAS, label_1_ind, label_2_RAS, label_2_ind, label_1_clusters, label_2_clusters, sub, subjects_dir, hemi, num_clusters: int = 1):
    """
    Among 2 labels, find the num_clusters closest clusters in each label according to the average path length between the 
    centroids of the cluster on the triangular mesh

    INPUT:
    label_1_RAS: np.array - array of RAS coordinates of label_1
    label_1_ind: np.array - array of indices of label_1
    label_2_RAS: np.array - array of RAS coordinates of label_2
    label_2_ind: np.array - array of indices of label_2
    label_1_clusters: np.array - array of cluster labels for label_1
    label_2_clusters: np.array - array of cluster labels for label_2
    num_clusters: int - number of clusters to return

    OUTPUT:
    closest_clusters: np.array - array of closest clusters
    """
    unique_clusters_1 = np.unique(label_1_clusters)
    unique_clusters_2 = np.unique(label_2_clusters)
    closest_clusters = []
    inflated_surface = nb.freesurfer.read_geometry(f'{subjects_dir}/{sub}/surf/{hemi}.inflated')
    for cluster1 in unique_clusters_1:
        for cluster2 in unique_clusters_2:
            cluster1_ind = label_1_ind[label_1_clusters == cluster1]
            cluster2_ind = label_2_ind[label_2_clusters == cluster2]
            cluster1_points = inflated_surface[0][cluster1_ind]
            cluster2_points = inflated_surface[0][cluster2_ind]
            dist = np.linalg.norm(np.mean(cluster1_points, axis=0) - np.mean(cluster2_points, axis=0))
            
            closest_clusters.append((cluster1, cluster2, dist))
    closest_clusters = np.array(closest_clusters)
    closest_clusters = closest_clusters[np.argsort(closest_clusters[:,2])]
    return closest_clusters[:num_clusters]



def plot_label_clusters(label_ind, clusters, subjects_dir, sub, hemi):
    """ 
    Plot the clustered label_RAS by cluster in interactive 3D

    INPUT:
    label_ind: np.array - array of indices of label
    clusters: np.array - array of clusters
    subjects_dir: str - path to subjects directory
    sub: str - subject ID
    hemi: str - hemisphere

    OUTPUT:
    plots the clusters
    """ 
    inflated_surface = nb.freesurfer.read_geometry(f'{subjects_dir}/{sub}/surf/{hemi}.inflated')
    inflated_RAS = inflated_surface[0][label_ind]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Specify a valid colormap (e.g., 'viridis', 'tab20', etc.)
    ax.scatter(inflated_RAS[:, 0], inflated_RAS[:, 1], inflated_RAS[:, 2], c=clusters, cmap='tab20')

    plt.show()
