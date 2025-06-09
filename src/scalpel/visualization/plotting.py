
from typing import List
import numpy as np
import trimesh as tm
from scalpel.utils import surface_utils
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
    if surface_type != 'wompwomp':
        scene = tm.Scene([mesh['gyrus'], mesh['sulcus']])
        apply_rotation(scene, view, hemi, reset=True)  # Reset to original position
    else:
        scene = tm.Scene([mesh['cortex']])
        apply_rotation(scene, view, hemi, reset=True)  # Reset to original position
    return scene

def plot_label(scene, ras_coords, faces, labels, label_name: str, view: str, hemi: str, face_colors=None, label_ind=None):
    ## Convenience method to plot a label on the surface
    if isinstance(face_colors, str):
        face_colors = DEFAULT_COLORS.get(face_colors.lower(), np.random.randint(0, 255, 3))
    if face_colors is None:
        face_colors = np.random.randint(0, 255, 3)
    if label_ind is None and label_name in labels:
        label_ind = labels[label_name].vertex_indexes
        face_colors = np.array(face_colors).astype(int)
        label_mesh = surface_utils.make_mesh(ras_coords, faces, label_ind, face_colors=face_colors, include_all = False)
    
    scene.add_geometry(label_mesh, geom_name=label_name)
    
    apply_rotation(scene, view, hemi, reset=True)
    
    return scene.show()

def plot(scene, hemi, view='lateral', labels: List[str] = None):
    # Reset the scene to the requested view
    apply_rotation(scene, view, hemi, reset=True)
    
    # Plot the surface
    if labels:
        for label in labels:
            plot_label(label, view=view)
    
    return scene.show()

def remove_label(scene, label_name: str):
    ## Remove label from surface
    if label_name not in scene.geometry.keys():
        raise ValueError(f"Label {label_name} not found in scene.")
    scene.delete_geometry(label_name)

def apply_rotation(object_mesh, view: str, hemi: str, reset: bool = False):
    """Apply rotation to mesh based on view and hemisphere.
    
    Parameters:
    -----------
    object_mesh : trimesh.Trimesh or trimesh.Scene
        The mesh or scene to transform
    view : str
        Desired view ('lateral', 'medial', 'ventral', 'dorsal')
    hemi : str
        Hemisphere ('lh' for left, 'rh' for right)
    reset : bool, default=False
        Whether to reset the mesh to its original position before applying new rotation
    """
    if reset:
        if isinstance(object_mesh, tm.Trimesh):
            if not hasattr(object_mesh, 'metadata'):
                object_mesh.metadata = {}
            if 'original_vertices' not in object_mesh.metadata:
                object_mesh.metadata['original_vertices'] = object_mesh.vertices.copy()
            else:
                object_mesh.vertices = object_mesh.metadata['original_vertices'].copy()
        elif hasattr(object_mesh, 'geometry'):
            for geom_name, geom in object_mesh.geometry.items():
                if not hasattr(geom, 'metadata'):
                    geom.metadata = {}
                if 'original_vertices' not in geom.metadata:
                    geom.metadata['original_vertices'] = geom.vertices.copy()
                else:
                    geom.vertices = geom.metadata['original_vertices'].copy()
    

    rotations = {
        ('lh', 'lateral'): [
            tm.transformations.rotation_matrix(np.pi/2, [0, 1, 0]),
            tm.transformations.rotation_matrix(np.pi/2, [0, 0, 1])
        ],
        ('rh', 'lateral'): [
            tm.transformations.rotation_matrix(-np.pi/2, [0, 1, 0]),
            tm.transformations.rotation_matrix(3*np.pi/2, [0, 0, 1])
        ],
        
        ('lh', 'medial'): [
            tm.transformations.rotation_matrix(-np.pi/2, [0, 1, 0]),
            tm.transformations.rotation_matrix(-np.pi/2, [0, 0, 1])
        ],
        ('rh', 'medial'): [
            tm.transformations.rotation_matrix(np.pi/2, [0, 1, 0]),
            tm.transformations.rotation_matrix(np.pi/2, [0, 0, 1])
        ],
        
        ('lh', 'ventral'): [
            tm.transformations.rotation_matrix(-np.pi, [0, 1, 0]),
            tm.transformations.rotation_matrix(0, [0, 0, 1]) 
        ],
        ('rh', 'ventral'): [
            tm.transformations.rotation_matrix(np.pi, [0, 1, 0]),
            tm.transformations.rotation_matrix(0, [0, 0, 1])
        ],
        
        ('lh', 'dorsal'): [
            tm.transformations.rotation_matrix(0, [1, 0, 0]),
            tm.transformations.rotation_matrix(0, [0, 0, 1])
        ],
        ('rh', 'dorsal'): [
            tm.transformations.rotation_matrix(0, [1, 0, 0]),
            tm.transformations.rotation_matrix(0, [0, 0, 1])
        ]
    }
    
    key = (hemi, view)
    if key in rotations:
        rotation_matrices = rotations[key]
        
        if not isinstance(rotation_matrices, list):
            rotation_matrices = [rotation_matrices]
        
        for rotation_matrix in rotation_matrices:
            if isinstance(object_mesh, tm.Trimesh):
                object_mesh.apply_transform(rotation_matrix)
            elif hasattr(object_mesh, 'geometry'):
                for geom_name, geom in object_mesh.geometry.items():
                    geom.apply_transform(rotation_matrix)
            else:
                raise ValueError("Object must be either a Trimesh or Scene")
    else:
        raise ValueError(f"Unsupported view/hemisphere combination: {view}/{hemi}")
    
def show_scene(scene):
    ## Plot the scene
    if scene:
        return scene.show()
    else:
        raise ValueError("Scene has not been initialized. Please call plot() first.")
