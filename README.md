![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

# CNL_scalpel

Scalpel is a Python library for analysis, segmentation, and plotting FreeSurfer cortical surface reconstructions.

## Features

- Simple object-oriented design for interacting with FreeSurfer subjects
- Interactive plotting in Jupyter notebooks
- Reading, editing, and writing FreeSurfer label files
- Label centroids, thresholding, boundary analysis, depth, and thickness measurements
- Gyral-sulcal analysis and clustering
- Surface area and cortical thickness calculations
- Euclidean distance measurements between labels

## Installation

### Requirements

- Python 3.10 or higher
- FreeSurfer installed locally ([installation guide](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall))
- FREESURFER_HOME environment variable defined and added to your PATH

### Setup

1. Clone the repository:
```bash
git clone https://github.com/b-parker/CNL_scalpel.git
cd CNL_scalpel
```

2. Create and activate a virtual environment:
```bash
conda create --name CNL_scalpel python=3.10
conda activate CNL_scalpel
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

```python
from scalpel.subject import ScalpelSubject

# Initialize a subject
subject = ScalpelSubject(
    subject_id="subj01",
    hemi="lh",
    subjects_dir="/path/to/subjects_dir",
    surface_type="inflated"
)
```

## Core Architecture

The ScalpelSubject class serves as the main interface and delegates specialized functionality to three core components:

- **ScalpelVisualizer**: Handles all visualization and plotting operations
- **ScalpelAnalyzer**: Performs surface analysis, clustering, and gyral-sulcal analysis
- **ScalpelMeasurer**: Calculates measurements like sulcal depth, surface area, and distances

This architecture maintains a simple user interface while organizing functionality logically.

## Basic Usage

### Accessing Surface Data

```python
# Basic properties
print(f"Subject ID: {subject.subject_id}")
print(f"Hemisphere: {subject.hemi}")
print(f"Surface type: {subject.surface_type}")

# Surface geometry
vertices = subject.surface_RAS  # RAS coordinates of vertices
faces = subject.faces          # Triangular faces of the mesh
vertex_indices = subject.vertex_indexes  # Unique vertex indices
```

### Working with Labels

Labels are regions of interest on the brain surface. The interface remains intuitive:

```python
# Load a label from FreeSurfer's label directory
subject.load_label('precentral')

# Load a custom label with vertex indices and coordinates
subject.load_label(
    'custom_region',
    label_idxs=vertex_indices,
    label_RAS=vertex_coordinates
)

# Access loaded labels
label = subject.labels['precentral']
vertices = label.vertex_indexes
coords = label.ras_coords

# Get label measurements (if stats file exists)
measurements = label.measurements
surface_area = label.get_measurement('total surface area (mm^2)')
```

## Visualization

The visualization system provides interactive plotting capabilities:

```python
# Plot the brain surface
subject.visualizer.plot(view='lateral')

# Plot with specific labels
subject.visualizer.plot(view='lateral', labels=['precentral', 'postcentral'])

# Plot individual labels with custom colors
subject.visualizer.plot_label('precentral', view='lateral', face_colors='red')

# Show the interactive scene
subject.visualizer.show()
```

Available views: `'lateral'`, `'medial'`, `'dorsal'`, `'ventral'`

![Lateral View](./assets/scalpel_lateral_inflated_plot.png)

### Plotting Labels

```python
# Load and plot a label
subject.load_label('IPS')
subject.visualizer.plot_label('IPS', view='lateral', face_colors='blue')
```

![IPS Label](./assets/scalpel_lateral_inflated_IPS_plot.png)

## Analysis and Measurements

### Sulcal Depth Analysis

```python
# Calculate sulcal depth (requires pial and gyral-inflated surfaces)
depth = subject.measurer.calculate_sulcal_depth(
    'sulcus_label',
    depth_pct=8,       # Percentage of deepest vertices
    n_deepest=100,     # Number of deepest vertices
    use_n_deepest=True # Use n_deepest vs percentage
)
print(f"Sulcal depth: {depth} mm")
```

### Surface Area and Thickness

```python
# Calculate surface area for a label
area = subject.measurer.calculate_surface_area('precentral')
print(f"Surface area: {area} mm²")

# Calculate mean cortical thickness
thickness = subject.measurer.calculate_cortical_thickness('precentral')
print(f"Mean thickness: {thickness} mm")
```

### Distance Measurements

```python
# Calculate Euclidean distance between label centroids
distance = subject.measurer.calculate_euclidean_distance(
    'label1', 'label2', method='centroid'
)
print(f"Distance between centroids: {distance} mm")

# Other distance methods: 'nearest', 'farthest'
```

## Advanced Analysis

### Gyral-Sulcal Analysis

```python
# Perform comprehensive sulcal-gyral relationship analysis
results = subject.analyzer.analyze_sulcal_gyral_relationships(
    'central_sulcus',
    gyral_clusters=300,
    sulcal_clusters=5,
    algorithm='kmeans',
    load_results=True  # Creates new labels for anterior/posterior gyri
)

# Access results
anterior_gyri = results['anterior_gyri']
posterior_gyri = results['posterior_gyri']
adjacency_map = results['adjacency_map']
```

### Gyral Gap Analysis

```python
# Find the gyral gap between two labels
gap_analysis = subject.analyzer.find_gyral_gap(
    'label1', 'label2',
    method='pca',
    n_clusters=[2, 3],
    load_label=True  # Creates shared gyral region label
)

# Access shared gyral region
shared_vertices = gap_analysis['shared_gyral_index']
shared_coords = gap_analysis['shared_gyral_ras']
```

### Label Thresholding

```python
# Threshold a label based on various measures
vertices, coords, values = subject.analyzer.threshold_label(
    'sulcus_label',
    threshold_type='percentile',    # 'percentile' or 'absolute'
    threshold_direction='>=',       # '>', '>=', '<', '<='
    threshold_value=90,             # 90th percentile
    threshold_measure='sulc',       # 'sulc', 'thickness', 'curv', 'label_stat'
    load_label=True,
    new_name='deep_sulcus'
)
```

### Clustering Analysis

```python
# Perform gyral clustering
clusters = subject.analyzer.perform_gyral_clustering(
    n_clusters=300,
    algorithm='kmeans'  # 'kmeans', 'agglomerative', 'dbscan'
)

# Find deepest sulci
deepest_indices = subject.analyzer.get_deepest_sulci(
    percentage=10,
    label_name='central_sulcus',  # Optional: within specific label
    load_label=True,
    result_label_name='deepest_central_sulcus'
)
```

## Complete Workflow Example

```python
from scalpel.subject import ScalpelSubject
import numpy as np

# Initialize subject
subject = ScalpelSubject("subj01", "lh", "/path/to/subjects_dir", "inflated")

# Load labels of interest
subject.load_label('precentral')
subject.load_label('postcentral')

# Visualize the labels
subject.visualizer.plot(view='lateral', labels=['precentral', 'postcentral'])

# Calculate measurements
pre_area = subject.measurer.calculate_surface_area('precentral')
post_area = subject.measurer.calculate_surface_area('postcentral')
distance = subject.measurer.calculate_euclidean_distance(
    'precentral', 'postcentral', method='centroid'
)

print(f"Precentral area: {pre_area} mm²")
print(f"Postcentral area: {post_area} mm²")
print(f"Distance between centroids: {distance} mm")

# Find the gyral gap between regions
gap_analysis = subject.analyzer.find_gyral_gap(
    'precentral', 'postcentral', load_label=True
)

# Analyze the central sulcus relationships
if 'central_sulcus' in subject.labels:
    depth = subject.measurer.calculate_sulcal_depth('central_sulcus')
    print(f"Central sulcus depth: {depth} mm")
    
    # Comprehensive gyral-sulcal analysis
    relationships = subject.analyzer.analyze_sulcal_gyral_relationships(
        'central_sulcus', load_results=True
    )

# Plot final results
subject.visualizer.plot(
    view='lateral', 
    labels=['precentral', 'postcentral', 'precentral_postcentral_shared_gyral']
)
subject.visualizer.show()
```

## Label Management

```python
# Combine multiple labels
combined_vertices = np.concatenate([
    subject.labels['label1'].vertex_indexes,
    subject.labels['label2'].vertex_indexes
])
combined_coords = np.concatenate([
    subject.labels['label1'].ras_coords,
    subject.labels['label2'].ras_coords
])
subject.load_label('combined_label', label_idxs=combined_vertices, label_RAS=combined_coords)

# Write labels to disk
subject.labels['custom_label'].write_label('my_custom_label')

# Calculate label centroid
centroid_coords = subject.analyzer.label_centroid('precentral', load=True)
```

## Tutorial

For a comprehensive tutorial with examples, see `tutorial.ipynb` in the repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
