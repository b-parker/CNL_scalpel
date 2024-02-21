import numpy as np
def get_thresholded_curv(curv: np.array, label_ind: np.array, threshold: float, suclal: bool = True):
    """
    Get the vertices of a mesh with curvature above a certain threshold

    INPUT:
    curv: np.array - array of curvature values (nb.freesurfer.read_morph_data(?h.curv))
    label_ind: np.array - array of indices of label
    threshold: float - threshold for curvature

    OUTPUT:
    thresholded_ind: np.array - array of indices of vertices with curvature above threshold

    Example:
    thresholded_ind = get_thresholded_curv(curv, inflated_gyrus_indexes, 0.05, suclal=False)
    """
    vertex_num = len(label_ind)
    threshold_number = vertex_num * threshold
    sorted_indexes = np.argsort(curv[label_ind])
    if not suclal:
        thresholded_ind = label_ind[sorted_indexes[:int(threshold_number)]]
    else:
        thresholded_ind = label_ind[sorted_indexes[-int(threshold_number):]]
    return thresholded_ind




def get_gyrus(label_index: np.array, label_RAS: np.array, curv: np.array, curv_threshold: int = 0):
    """ 
    Returns all label indices and RAS coordinates for gyrus within freesurfer label

    INPUT:
    _____
    label_index: np.array - numpy array of label indexes from src.read_label()
    label_RAS: np.array - numpy array of label RAS vertices from src.read_label()
    curv: np.array - numpy array of curvature values from nb.freesurfer.read_morph_data()
    curv_threshold: int - value for thresholding curvature value

    OUTPUT:
    gyrus_index: np.array - numpy array of gyrus indexes from src.read_label()
    gyrus_RAS: np.array - numpy array of gyrus RAS vertices from src.read_label()

    """
        
    gyrus_index = []
    gyrus_RAS = []

    for point, RAS in zip(label_index, label_RAS):
        if curv[point] < curv_threshold:
            gyrus_index.append(point)
            gyrus_RAS.append(RAS)
        else:
            continue
    return np.array(gyrus_index), np.array(gyrus_RAS)


def get_sulcus(label_index: np.array, label_RAS: np.array, curv: np.array, curv_threshold: int = 0):
    """ 
    Returns all label indices and RAS coordinates for sulcus within freesurfer label

    INPUT:
    _____
    label_index: np.array - numpy array of label indexes from src.read_label()
    label_RAS: np.array - numpy array of label RAS vertices from src.read_label()
    curv: np.array - numpy array of curvature values from nb.freesurfer.read_morph_data()
    curv_threshold: int - value for thresholding curvature value

    OUTPUT:
    sulcus_index: np.array - numpy array of sulcus indexes from src.read_label()
    sulcus_RAS: np.array - numpy array of sulcus RAS vertices from src.read_label()

    """
        
    sulcus_index = []
    sulcus_RAS = []

    for point, RAS in zip(label_index, label_RAS):
        if not isinstance(point, int):
            point = int(point)

        if curv[point] > curv_threshold:
            sulcus_index.append(point)
            sulcus_RAS.append(RAS)
        else:
            continue
    return np.array(sulcus_index), np.array(sulcus_RAS)