import numpy as np
import trimesh as tm

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