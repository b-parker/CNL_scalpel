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
from scalpel.utils.freesurfer_utils import *


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
    """
    # Convert to set for O(1) lookup instead of O(n)
    label_set = set(label_ind)
    
    all_label_faces = []
    if include_all == False:
        for face in faces:
            if all(point in label_set for point in face):
                all_label_faces.append(face)
    else:
        for face in faces:
            if any(point in label_set for point in face):
                all_label_faces.append(face)
    return np.array(all_label_faces)
        
