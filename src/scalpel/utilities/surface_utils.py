# Utilities
from pathlib import Path
import os, sys
import subprocess as sp
from functools import partial 
from time import time
from typing import List, Dict, Tuple


# Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# Brain
import nibabel as nb

# Plotting1=--0
import matplotlib.pyplot as plt

# Meshes
import trimesh as tm
import networkx as nx
#import meshplot 
from scalpel.utilities.freesurfer_utils import *






############################################################################################################
############################################################################################################
############################################################################################################


#                   Geodesic


##########################################################################################################
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
 
    dist_mat = np.zeros((n_labels,n_labels))

    
    for r1 in np.arange(n_labels):
        #print('r1',r1,label_inds_all[r1])
        for r2 in np.arange(n_labels):
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

def cluster_label_KMeans(label_ind, points, n_clusters: int = 2):
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

def cluster_label_mean_shift(label_ind, points, bandwidth: float = 1.5):
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



def find_adjacent_indices(label_ind, faces):
    """
    Find all adjacent indices in a label

    INPUT:
    label_ind: np.array - array of indices of label
    faces: np.array - array of faces in mesh

    OUTPUT:
    adj_ind: np.array - array of adjacent indices
    """
    adj_ind = np.array([])
    for ind in label_ind:
        adj = np.unique(faces[np.where(faces == ind)[0]].flatten())
        adj_ind = np.append(adj_ind, adj)
    adj_ind = np.unique(adj_ind)
    return adj_ind


### combine labels

def combine_labels(subject: "ScalpelSubject", labels: List[str], save_to_subject: bool = False):
    """ 
    Combine labels into a single label

    INPUT:
    subject: ScalpelSubject - subject object
    labels: List[str] - list of labels to combine

    OUTPUT:
    combined_label_ind: np.array - array of indices of combined label
    combined_label_RAS: np.array - array of RAS coordinates of combined label
    
    """
    combined_label_ind = np.hstack(([subject.labels[label][0] for label in labels]))
    combined_label_RAS = subject.ras_coords[combined_label_ind]
    if save_to_subject:
        subject.labels[f'combined_{"-".join(labels)}'] = [combined_label_ind, combined_label_RAS]
    return combined_label_ind, combined_label_RAS

### PCA
def pca_label(subject: "ScalpelSubject", labels: List[str], n_components: int = 2):
    ## PCA on these labels
    # Step 1: Standardize the data
    scaler = StandardScaler()
    label_12_faces = get_faces_from_vertices(subject.faces, subject.labels[combined_label][0])
    label_12_boundary = find_label_boundary(label_12_faces)
    label_12_boundary_RAS = subject.ras_coords[label_12_boundary]
    points_scaled = scaler.fit_transform(label_12_boundary_RAS)

    # Step 2: Apply PCA
    pca = PCA(n_components=2)  
    points_pca = pca.fit_transform(points_scaled)

    points_for_clustering = points_pca[:, 0].reshape(-1, 1)
    label1_points = points_for_clustering[np.isin(label_12_boundary, subject.labels['IPS'][0])]
    label2_points = points_for_clustering[np.isin(label_12_boundary, subject.labels['combined_i'][0])]

    Ag_clust_1 = AgglomerativeClustering(n_clusters=2).fit_predict(label1_points)
    Ag_clust_2 = AgglomerativeClustering(n_clusters=3).fit_predict(label2_points)
    clusters2_adjusted = Ag_clust_2 + 5
    Ag_clust_3 = AgglomerativeClustering(n_clusters=5).fit_predict(points_for_clustering)


    

############################################################################################################
############################################################################################################
############################################################################################################


                #   Morphometric Parcellation


############################################################################################################
############################################################################################################
############################################################################################################

def get_thresholded_curv(curv: np.array, label_ind: np.array, threshold: float, sulcal: bool = True):
    """
    Get the vertices of a mesh with curvature above a certain threshold

    INPUT:
    curv: np.array - array of curvature values (nb.freesurfer.read_morph_data(?h.curv))
    label_ind: np.array - array of indices of label
    threshold: float - threshold for curvature

    OUTPUT:
    thresholded_ind: np.array - array of indices of vertices with curvature above threshold
    """
    vertex_num = len(label_ind)
    threshold_number = vertex_num * threshold
    sorted_indexes = np.argsort(curv[label_ind])
    if not sulcal:
        thresholded_ind = label_ind[sorted_indexes[:int(threshold_number)]]
    else:
        thresholded_ind = label_ind[sorted_indexes[-int(threshold_number):]]
    return thresholded_ind

def get_thresholded_thickness(curv: np.array, label_ind: np.array, threshold: float):
    """
    Get the vertices of a mesh with curvature above a certain threshold

    INPUT:
    curv: np.array - array of curvature values (nb.freesurfer.read_morph_data(?h.curv))
    label_ind: np.array - array of indices of label
    threshold: float - threshold for curvature

    OUTPUT:
    thresholded_ind: np.array - array of indices of vertices with curvature above threshold
    """
    vertex_num = len(label_ind)
    threshold_number = vertex_num * threshold
    sorted_indexes = np.argsort(curv[label_ind]) 

    thresholded_ind = label_ind[sorted_indexes[-int(threshold_number):]]
    return thresholded_ind   


def calculate_geometric_centroid(vertices, faces):
    """
    Calculate the geometric centroid of a triangular mesh.
    
    Parameters:
        vertices: numpy array of shape (N, 3) containing vertex coordinates
        faces: numpy array of shape (M, 3) containing vertex indices for each triangle
    
    Returns:
        centroid: numpy array of shape (3,) containing the x,y,z coordinates of the geometric centroid
    """

    triangle_centroids = np.zeros((len(faces), 3))
    triangle_areas = np.zeros(len(faces))
    
    for i, face in enumerate(faces):
        # Get vertices for this triangle
        triangle_vertices = vertices[face]
        
        # Calculate triangle centroid (average of three vertices)
        triangle_centroids[i] = np.mean(triangle_vertices, axis=0)
        
        # Calculate triangle area using cross product
        # Area = 0.5 * ||(v2-v1) × (v3-v1)||
        v1, v2, v3 = triangle_vertices
        cross_product = np.cross(v2 - v1, v3 - v1)
        triangle_areas[i] = 0.5 * np.linalg.norm(cross_product)
    
    # Calculate weighted centroid using triangle areas as weights
    total_area = np.sum(triangle_areas)
    if total_area == 0:
        raise ValueError("Total mesh area is zero")
        
    centroid = np.sum(triangle_centroids * triangle_areas[:, np.newaxis], axis=0) / total_area
    
    return centroid

def find_closest_vertex(centroid, vertices):
    """
    Find the vertex closest to the calculated centroid.
    
    Parameters:
        centroid: numpy array of shape (3,) containing centroid coordinates
        vertices: numpy array of shape (N, 3) containing vertex coordinates
    
    Returns:
        index: index of the closest vertex
        distance: distance to the closest vertex
    """
    distances = np.linalg.norm(vertices - centroid, axis=1)
    closest_idx = np.argmin(distances)
    return closest_idx, distances[closest_idx]

def make_mesh(inflated_points: np.array, faces: np.array, label_ind: np.array, **kwargs) -> tm.Trimesh:
    """ 
    Given a set of indices, construct a mesh of the vertices in the indices along a surface

    INPUT: 
    faces: np.array - array of faces in mesh
    label_ind: np.array - array of indices of label

    OUTPUT:
    label_mesh: tm.Trimesh - mesh of label
    """
    if 'include_all' in kwargs:
        include_all = kwargs['include_all']
    else:
        include_all = False

    label_faces = get_faces_from_vertices(faces, label_ind, include_all=include_all)
    label_mesh = tm.Trimesh(vertices=inflated_points, faces=label_faces, process=False, face_colors=kwargs['face_colors'])
    return label_mesh

def get_faces_from_vertices(faces : np.array, label_ind : np.array, include_all : bool = False):
    """
    Takes a list of faces and label indices
    Returns the faces that contain the indices

    INPUT:
    faces: array of faces composed of 3 points
    label_ind: array of indices of points in the label (first colum of label file; 0 index in read_label)
    include_all: bool - if True, return faces that contain any of the points in the label

    OUTPUT:
    label_faces: array of faces that contain the points in the label
    """
    all_label_faces = []
    if include_all == False:
        for face in faces:
            if all([point in label_ind for point in face]):
                all_label_faces.append(face)
    else:
        for face in faces:
            if any([point in label_ind for point in face]):
                all_label_faces.append(face)
    return np.array(all_label_faces)


        


    
    

        

