
from typing import List
import numpy as np
import trimesh as tm
from src.utilities import geometry_utils
NoneType = type(None)

# Define a dictionary of default colors
DEFAULT_COLORS = {
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'white': [255, 255, 255],
    'gray': [128, 128, 128],
    'yellow': [255, 255, 0],
    'cyan': [0, 255, 255],
    'magenta': [255, 0, 255],
    'black': [0, 0, 0],
    'orange': [255, 165, 0],
    'purple': [128, 0, 128],
    'brown': [165, 42, 42],
    'pink': [255, 192, 203],
    'lime': [0, 255, 0],
    'navy': [0, 0, 128],
    'teal': [0, 128, 128],
    'lavender': [230, 230, 250],
    'gold': [255, 215, 0],
    'salmon': [250, 128, 114],
    'olive': [128, 128, 0],
    'maroon': [128, 0, 0],
    'skyblue': [135, 206, 235],
    'chocolate': [210, 105, 30],
    'crimson': [220, 20, 60],
    'coral': [255, 127, 80],
    'indigo': [75, 0, 130],
    'violet': [238, 130, 238],
    'khaki': [240, 230, 140],
    'plum': [221, 160, 221],
    'orchid': [218, 112, 214]
}

def initialize_scene(mesh, view: str, hemi: str, surface_type: str):
    scene = None
    if surface_type == 'inflated':
        scene = tm.Scene([mesh['gyrus'], mesh['sulcus']])
        apply_rotation(scene, view, hemi)
    else:
        scene = tm.Scene([mesh['cortex']])
        apply_rotation(scene, view, hemi)
    return scene

def plot(scene, view='lateral', labels: List[str] = None, plot_label_func=None):
    ## Plot the surface
    if labels:
        for label in labels:
            plot_label_func(label, view = view)
    return scene.show()

def plot_label(scene, ras_coords, faces, labels, label_name: str, view: str, hemi: str, face_colors=None, label_ind=None):
    ## Convenience method to plot a label on the surface
    if isinstance(face_colors, str):
        face_colors = DEFAULT_COLORS.get(face_colors.lower(), np.random.randint(0, 255, 3))
    if face_colors is None:
        face_colors = np.random.randint(0, 255, 3)
    if label_ind is None and label_name in labels:
        label_ind = labels[label_name].vertex_indexes

    face_colors = np.array(face_colors).astype(int)
    label_mesh = geometry_utils.make_mesh(ras_coords, faces, label_ind, face_colors=face_colors)
    apply_rotation(label_mesh, view, hemi)
    
    scene.add_geometry(label_mesh, geom_name=label_name)
    return scene.show()

def remove_label(scene, label_name: str):
    ## Remove label from surface
    if label_name not in scene.geometry.keys():
        raise ValueError(f"Label {label_name} not found in scene.")
    scene.delete_geometry(label_name)

def apply_rotation(object_mesh, view: str, hemi: str):
    if (hemi == 'lh' and view == 'lateral') or (hemi == 'rh' and view == 'medial'):
        object_mesh.apply_transform(tm.transformations.rotation_matrix(np.pi/2, [0, 0, 1]))
        object_mesh.apply_transform(tm.transformations.rotation_matrix(3*np.pi/2, [1, 0, 0]))
    if (hemi == 'lh' and view == 'medial') or (hemi == 'rh' and view == 'lateral'):
        object_mesh.apply_transform(tm.transformations.rotation_matrix(-np.pi/2, [0, 0, 1]))
        object_mesh.apply_transform(tm.transformations.rotation_matrix(3*np.pi/2, [1, 0, 0]))

def show_scene(scene):
    ## Plot the scene
    if scene:
        return scene.show()
    else:
        raise ValueError("Scene has not been initialized. Please call plot() first.")
