import numpy as np 

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
