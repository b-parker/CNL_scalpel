import numpy as np 
import pandas as pd
import nibabel as nb




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



"""
Example Usage:

import nibabel as nb
import numpy as np
import pandas as pd

subject_dir = "/Users/benparker/Desktop/cnl/subjects/100206"
label_ind, ras = read_label(f"{subject_dir}/label/lh.IPS.label")

all_ras, all_faces = nib.freesurfer.read_geometry(f"{subject}/surf/lh.inflated")

label_faces = get_faces_from_vertices(all_faces, label_ind)
label_subset_list = get_label_subsets(label_faces, all_faces)

"""