from __future__ import annotations
from typing import List, Tuple, Dict, Union, TYPE_CHECKING, Optional
import numpy as np
import trimesh as tm
from pathlib import Path
import io
from PIL import Image

# Import the plotting utilities (which will remain in the original plotting.py)
from scalpel.visualization.plotting import (
    initialize_scene, 
    plot as plot_util, 
    plot_label as plot_label_util, 
    remove_label as remove_label_util, 
    show_scene, 
    apply_rotation,
    DEFAULT_COLORS
)

# Import utilities for mesh handling
from scalpel.utils import surface_utils

if TYPE_CHECKING:
    from scalpel.subject import ScalpelSubject

class ScalpelVisualizer:
    """
    Class for visualizing brain surface data.
    
    This class handles all visualization-related tasks for a ScalpelSubject,
    including plotting the cortical surface, labels, and saving images.
    """
    
    def __init__(self, subject: 'ScalpelSubject'):
        """
        Initialize a ScalpelVisualizer.
        
        Parameters:
        -----------
        subject : ScalpelSubject
            The subject to visualize
        """
        self._subject = subject
        self._scene = None
        self._mesh = {}
        
    @property
    def subject(self):
        """Get the associated ScalpelSubject."""
        return self._subject
        
    @property
    def scene(self):
        """Get the current scene. Creates a scene if one doesn't exist."""
        if self._scene is None:
            self._scene = self._create_scene()
        return self._scene
    
    @property
    def mesh(self):
        """Build and get the cortical mesh."""
        if not self._mesh:
            self._build_mesh()
        return self._mesh
    
    def _build_mesh(self):
        """
        Build the cortical mesh for visualization.
        
        Creates separate meshes for gyri and sulci with appropriate colors.
        """
        # Set default colors
        gyrus_gray = [250, 250, 250]
        sulcus_gray = [130, 130, 130]
        
        # Access subject properties
        ras_coords = self._subject.surface_RAS
        faces = self._subject.faces
        
        # Get gyrus and sulcus information from the subject's analyzer
        gyrus = self._subject.analyzer.gyrus
        sulcus = self._subject.analyzer.sulcus
        
        print('Initial plot builds cortical mesh (~1 minute)')
        
        # Create meshes for gyri and sulci
        gyrus_mesh = surface_utils.make_mesh(
            ras_coords, faces, gyrus[0], face_colors=gyrus_gray
        )
        sulcus_mesh = surface_utils.make_mesh(
            ras_coords, faces, sulcus[0], face_colors=sulcus_gray, include_all=True
        )
        
        # Store the meshes
        self._mesh['gyrus'] = gyrus_mesh
        self._mesh['sulcus'] = sulcus_mesh
        
        return self._mesh
    
    def _create_scene(self):
        """Create a new scene with the cortical mesh."""
        mesh = self.mesh
        return initialize_scene(
            mesh, 
            view='lateral', 
            hemi=self._subject.hemi, 
            surface_type='inflated'
        )
    
    def plot(self, view: str = 'lateral', labels: List[str] = None):
        """
        Plot the cortical surface, optionally with labels.
        
        Parameters:
        -----------
        view : str, default='lateral'
            View angle ('lateral', 'medial', 'ventral', 'dorsal')
        labels : List[str], optional
            Names of labels to plot
            
        Returns:
        --------
        trimesh.Scene
            The visualized scene
        """
        # Get or create scene
        scene = self.scene
        
        # Apply the requested view
        apply_rotation(
            scene, 
            view=view, 
            hemi=self._subject.hemi, 
            reset=True
        )
        
        # Add labels if specified
        if labels:
            for label_name in labels:
                self.plot_label(label_name, view=view)
        
        # Show the scene
        return scene.show()
    
    def plot_label(
        self, 
        label_name: str, 
        view: str = 'lateral', 
        label_ind: np.ndarray = None, 
        face_colors: Union[str, List[int], np.ndarray] = None
    ):
        """
        Plot a label on the cortical surface.
        
        Parameters:
        -----------
        label_name : str
            Name of the label to plot
        view : str, default='lateral'
            View angle ('lateral', 'medial', 'ventral', 'dorsal')
        label_ind : np.ndarray, optional
            Vertex indices for the label, if not using a stored label
        face_colors : Union[str, List[int], np.ndarray], optional
            Colors for the label faces
            
        Returns:
        --------
        trimesh.Scene
            The visualized scene
        """
        # Ensure the subject has this label
        if label_ind is None:
            assert label_name in self._subject.labels, f"Label {label_name} not found in subject {self._subject.subject_id}"
        
        # Get or create scene
        scene = self.scene
        
        # Process face colors
        if isinstance(face_colors, str):
            face_colors = DEFAULT_COLORS.get(face_colors.lower(), np.random.randint(0, 255, 3))
        if face_colors is None:
            face_colors = np.random.randint(0, 255, 3)
        
        # Get vertices and create mesh
        if label_ind is None and label_name in self._subject.labels:
            label_ind = self._subject.labels[label_name].vertex_indexes
        
        face_colors = np.array(face_colors).astype(int)
        
        # Create the label mesh
        label_mesh = surface_utils.make_mesh(
            self._subject.surface_RAS, 
            self._subject.faces, 
            label_ind, 
            face_colors=face_colors, 
            include_all=False
        )
        
        # Add to scene
        scene.add_geometry(label_mesh, geom_name=label_name)
        
        # Apply the view
        apply_rotation(scene, view, self._subject.hemi, reset=True)
        
        # Show the scene
        return scene.show()
    
    def remove_label(self, label_name: str):
        """
        Remove a label from the visualization.
        
        Parameters:
        -----------
        label_name : str
            Name of the label to remove
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if self._scene is None:
            return False
            
        if label_name not in self._scene.geometry.keys():
            raise ValueError(f"Label {label_name} not found in scene.")
        
        # Remove from scene
        self._scene.delete_geometry(label_name)
        return True
    
    def show(self):
        """
        Show the current scene.
        
        Returns:
        --------
        trimesh.Scene
            The visualized scene
        """
        if self._scene is None:
            self._scene = self._create_scene()
        
        return self._scene.show()
    
    def save_plot(
        self, 
        filename: str, 
        save_dir: Union[str, Path] = None, 
        distance: int = 500, 
        resolution: Union[str, Tuple[int, int]] = 'low'
    ):
        """
        Save the current scene as an image.
        
        Parameters:
        -----------
        filename : str
            Name of the output file
        save_dir : Union[str, Path], optional
            Directory to save the file in
        distance : int, default=500
            Camera distance
        resolution : Union[str, Tuple[int, int]], default='low'
            Image resolution, either a string ('low', 'medium', 'high')
            or a tuple of (width, height)
            
        Returns:
        --------
        None
        
        Raises:
        -------
        ValueError
            If the scene is not initialized or if parameters are invalid
        """
        if self._scene is None:
            raise ValueError("Scene not initialized. Please call plot() first.")
        
        # Process resolution
        if isinstance(resolution, str):
            resolution_map = {
                'low': (512, 512),
                'medium': (720, 720),
                'high': (1080, 1080)
            }
            if resolution not in resolution_map:
                raise ValueError(f"Invalid resolution '{resolution}'. Must be one of {list(resolution_map.keys())}")
            resolution = resolution_map[resolution]
        
        # Set camera distance
        self._scene.set_camera(distance=distance)
        
        # Prepare filename
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir / filename
        
        # Save image
        data = self._scene.save_image(resolution=resolution)
        image = Image.open(io.BytesIO(data))
        image.save(filename)
        print(f"Plot saved to {filename}")