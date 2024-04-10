
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

def write_label(label_indexes: np.array, label_RAS: np.array, label_name: str, hemi: str, subject_dir: str or Path, surface_type: str = 'white', overwrite: bool = False):
    """
    Write freesurfer label file from label indexes and RAS coordinates

    INPUT:
    _____
    label_indexes: np.array - numpy array of label indexes from src.read_label()
    label_RAS: np.array - numpy array of label RAS vertices from src.read_label()
    label_name: str - name of label
    hemi: str - hemisphere of label
    subject_dir: str or Path - path to subject directory
    surface_type: str - surface type which label_RAS come from ['white', 'pial', 'orig']
    
    """
    
    if isinstance(subject_dir, str):
        subject_dir = Path(subject_dir)
    

    label_filename = subject_dir / 'label' / f'{hemi}.{label_name}.label'
    
    if overwrite == False:
        assert not label_filename.exists(), f"{hemi}.{label_name} already exists for subject at {subject_dir.absolute()}"

    subject_id = subject_dir.name
    label_length = label_indexes.shape[0]

    print(f'Writing label {label_filename.name} for {subject_id}')
    
    with open(label_filename.absolute(), 'w') as label_file:
        label_file.writelines(f'#!ascii label  , from subject {subject_id} vox2ras=TkReg coords={surface_type}\n')
        label_file.writelines(f'{label_length}\n')
        for i in range(label_length):
            label_line = f"{label_indexes[i]} {label_RAS[i][0]} {label_RAS[i][1]} {label_RAS[i][2]} 0.0000000000 \n"
            label_file.write(label_line)
