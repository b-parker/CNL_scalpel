""" System imports """
from pathlib import Path
import os
import subprocess as sp
import shlex
import tarfile
import json
import datetime
from numpy.random import randint
import numpy as np
import pandas as pd

from typing import Union, Optional, List, Tuple

try:
    from typing import NoneType
except ImportError:
    NoneType = type(None)
    
def freesurfer_label2label(
    src_subject: str,
    src_label: str,
    trg_subject: str,
    trg_label: str,
    reg_method: str,
    hemi: Optional[str] = None,
    src_hemi: Optional[str] = None,
    trg_hemi: Optional[str] = None,
    src_ico_order: Optional[int] = None,
    trg_ico_order: Optional[int] = None,
    trg_surf: Optional[str] = None,
    surf_reg: Optional[str] = None,
    src_surf_reg: Optional[str] = None,
    trg_surf_reg: Optional[str] = None,
    src_mask: Optional[str] = None,
    src_mask_sign: Optional[str] = None,
    src_mask_frame: Optional[int] = None,
    proj_abs: Optional[float] = None,
    proj_frac: Optional[float] = None,
    subjects_dir: Optional[str] = None,
    no_hash: bool = False,
    no_rev_map: bool = False,
    freesurfer_home: Optional[str] = None,
    debug: bool = False
) -> sp.CompletedProcess:
    """
    Converts a label in one subject's space to a label in another subject's space.
    
    Args:
        src_subject: Source subject name
        src_label: Source label filename
        trg_subject: Target subject name
        trg_label: Target label filename
        reg_method: Registration method ('surface' or 'volume')
        hemi: Hemisphere ('lh' or 'rh'), required with surface reg_method
        src_hemi: Source hemisphere (if different from hemi)
        trg_hemi: Target hemisphere (if different from hemi)
        src_ico_order: Source icosahedron order (when src_subject='ico')
        trg_ico_order: Target icosahedron order (when trg_subject='ico')
        trg_surf: Get xyz from this surface (default: 'white')
        surf_reg: Surface registration file (default: 'sphere.reg')
        src_surf_reg: Source surface registration file
        trg_surf_reg: Target surface registration file
        src_mask: Source mask surface value file
        src_mask_sign: Source mask sign ('abs', 'pos', 'neg')
        src_mask_frame: Source mask frame number (0-based)
        proj_abs: Project absolute distance (mm) along surface normal
        proj_frac: Project fraction of thickness along surface normal
        subjects_dir: FreeSurfer subjects directory
        no_hash: Don't use hash table when reg_method is surface
        no_rev_map: Don't use reverse mapping when reg_method is surface
        freesurfer_home: FreeSurfer installation directory
        debug: Print debug information
        
    Returns:
        CompletedProcess object with command results
    """
    # Validate inputs
    if reg_method not in ['surface', 'volume']:
        raise ValueError(f"Registration method must be 'surface' or 'volume', got {reg_method}")
    
    if reg_method == 'surface' and not hemi:
        raise ValueError("Hemisphere (--hemi) is required when using surface registration method")
    
    if hemi and hemi not in ['lh', 'rh']:
        raise ValueError(f"Hemisphere must be 'lh' or 'rh', got {hemi}")
    
    if src_hemi and src_hemi not in ['lh', 'rh']:
        raise ValueError(f"Source hemisphere must be 'lh' or 'rh', got {src_hemi}")
    
    if trg_hemi and trg_hemi not in ['lh', 'rh']:
        raise ValueError(f"Target hemisphere must be 'lh' or 'rh', got {trg_hemi}")
    
    if src_subject == 'ico' and src_ico_order is None:
        raise ValueError("Source icosahedron order (--srcicoorder) is required when src_subject is 'ico'")
    
    if trg_subject == 'ico' and trg_ico_order is None:
        raise ValueError("Target icosahedron order (--trgicoorder) is required when trg_subject is 'ico'")
    
    # Find FreeSurfer home
    if freesurfer_home is None:
        freesurfer_home = os.environ.get('FREESURFER_HOME')
        if not freesurfer_home:
            # Common FreeSurfer installation paths
            possible_paths = [
                '/usr/local/freesurfer',
                '/opt/freesurfer',
                '/home/freesurfer',
                '/usr/share/freesurfer'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    freesurfer_home = path
                    break
            if not freesurfer_home:
                raise ValueError("FREESURFER_HOME not found. Please set it manually.")
    
    # Set up environment
    env = os.environ.copy()
    if subjects_dir:
        env['SUBJECTS_DIR'] = str(Path(subjects_dir).absolute())
    elif 'SUBJECTS_DIR' not in env:
        raise ValueError("SUBJECTS_DIR must be specified either as an argument or environment variable")
    
    env['FREESURFER_HOME'] = freesurfer_home
    
    # FreeSurfer paths
    fs_bin = os.path.join(freesurfer_home, 'bin')
    fs_lib = os.path.join(freesurfer_home, 'lib')
    
    # Update PATH and LD_LIBRARY_PATH
    env['PATH'] = f"{fs_bin}:{env.get('PATH', '')}"
    if 'LD_LIBRARY_PATH' in env:
        env['LD_LIBRARY_PATH'] = f"{fs_lib}:{env['LD_LIBRARY_PATH']}"
    else:
        env['LD_LIBRARY_PATH'] = fs_lib
    
    # Print debug info
    if debug:
        print(f"Using FREESURFER_HOME: {env['FREESURFER_HOME']}")
        print(f"Using SUBJECTS_DIR: {env['SUBJECTS_DIR']}")
    
    # Build command
    cmd = ['mri_label2label']
    
    # Required arguments
    cmd.extend(['--srcsubject', src_subject])
    cmd.extend(['--srclabel', src_label])
    cmd.extend(['--trgsubject', trg_subject])
    cmd.extend(['--trglabel', trg_label])
    cmd.extend(['--regmethod', reg_method])
    
    # Required for surface registration
    if hemi:
        cmd.extend(['--hemi', hemi])
    
    # Optional arguments
    if src_hemi:
        cmd.extend(['--srchemi', src_hemi])
    
    if trg_hemi:
        cmd.extend(['--trghemi', trg_hemi])
    
    if src_ico_order is not None:
        cmd.extend(['--srcicoorder', str(src_ico_order)])
    
    if trg_ico_order is not None:
        cmd.extend(['--trgicoorder', str(trg_ico_order)])
    
    if trg_surf:
        cmd.extend(['--trgsurf', trg_surf])
    
    if surf_reg:
        cmd.extend(['--surfreg', surf_reg])
    
    if src_surf_reg:
        cmd.extend(['--srcsurfreg', src_surf_reg])
    
    if trg_surf_reg:
        cmd.extend(['--trgsurfreg', trg_surf_reg])
    
    if src_mask:
        cmd.extend(['--srcmask', src_mask])
    
    if src_mask_sign:
        cmd.extend(['--srcmasksign', src_mask_sign])
    
    if src_mask_frame is not None:
        cmd.extend(['--srcmaskframe', str(src_mask_frame)])
    
    if proj_abs is not None:
        cmd.extend(['--projabs', str(proj_abs)])
    
    if proj_frac is not None:
        cmd.extend(['--projfrac', str(proj_frac)])
    
    if subjects_dir:
        cmd.extend(['--sd', subjects_dir])
    
    if no_hash:
        cmd.append('--nohash')
    
    if no_rev_map:
        cmd.append('--norevmap')
    
    # Log the command
    cmd_str = ' '.join(cmd)
    if debug:
        print(f"Running command: {cmd_str}")
    
    # Execute command with error handling
    try:
        result = sp.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            env=env
        )
        if debug:
            print("Command succeeded")
            print(f"Output: {result.stdout}")
        return result
        
    except sp.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        print(f"Standard output: {e.stdout}")
        raise

    
def freesurfer_mris_label2annot(
    subject_id: str,
    hemi: str,
    annot_name: str,
    subjects_dir: Optional[str] = None,
    ctab_file: Optional[str] = None,
    label_files: Optional[List[str]] = None,
    label_dir: Optional[str] = None,
    use_default_label_dir: bool = False,
    no_unknown: bool = False,
    nhits_file: Optional[str] = None,
    offset: Optional[int] = None,
    max_stat_winner: bool = False,
    threshold: Optional[float] = None,
    surface: str = 'orig',
    no_verbose: bool = False,
    debug: bool = False,
    freesurfer_home: Optional[str] = None
) -> sp.CompletedProcess:
    """
    Runs FreeSurfer's mris_label2annot command to amalgamate label files into an annotation file.
    
    Args:
        subject_id: FreeSurfer subject ID
        hemi: Hemisphere ('lh' or 'rh')
        annot_name: Output annotation name (will be saved as hemi.annot_name.annot)
        subjects_dir: FreeSurfer subjects directory (if None, uses env variable)
        ctab_file: Color table file defining structure names, indices, and colors
        label_files: List of label files to include in the annotation
        label_dir: Directory to look for label files when using ctab
        use_default_label_dir: Use subject's default label directory
        no_unknown: Start label numbering at index 0 instead of 1
        nhits_file: Output file showing number of labels assigned to each vertex
        offset: Value to add to label number to get CTAB index
        max_stat_winner: Keep label with highest 'stat' value
        threshold: Threshold label by stats field
        surface: Surface name to use (default: 'orig')
        no_verbose: Turn off overlap and stat override messages
        debug: Enable debug mode
        freesurfer_home: FreeSurfer installation directory
        
    Returns:
        CompletedProcess object with command results
    """
    # Validate inputs
    if hemi not in ['lh', 'rh']:
        raise ValueError(f"Hemisphere must be 'lh' or 'rh', got {hemi}")
    
    # Find FreeSurfer home
    if freesurfer_home is None:
        freesurfer_home = os.environ.get('FREESURFER_HOME')
        if not freesurfer_home:
            # Common FreeSurfer installation paths
            possible_paths = [
                '/usr/local/freesurfer',
                '/opt/freesurfer',
                '/home/freesurfer',
                '/usr/share/freesurfer'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    freesurfer_home = path
                    break
            if not freesurfer_home:
                raise ValueError("FREESURFER_HOME not found. Please set it manually.")
    
    # Set up environment
    env = os.environ.copy()
    if subjects_dir:
        env['SUBJECTS_DIR'] = str(Path(subjects_dir).absolute())
    elif 'SUBJECTS_DIR' not in env:
        raise ValueError("SUBJECTS_DIR must be specified either as an argument or environment variable")
    
    env['FREESURFER_HOME'] = freesurfer_home
    
    # FreeSurfer paths
    fs_bin = os.path.join(freesurfer_home, 'bin')
    fs_lib = os.path.join(freesurfer_home, 'lib')
    
    # Update PATH and LD_LIBRARY_PATH
    env['PATH'] = f"{fs_bin}:{env.get('PATH', '')}"
    if 'LD_LIBRARY_PATH' in env:
        env['LD_LIBRARY_PATH'] = f"{fs_lib}:{env['LD_LIBRARY_PATH']}"
    else:
        env['LD_LIBRARY_PATH'] = fs_lib
    
    # Print debug info
    if debug:
        print(f"Using FREESURFER_HOME: {env['FREESURFER_HOME']}")
        print(f"Using SUBJECTS_DIR: {env['SUBJECTS_DIR']}")
    
    # Build command
    cmd = ['mris_label2annot']
    
    # Required arguments
    cmd.extend(['--subject', subject_id])
    cmd.extend(['--hemi', hemi])
    cmd.extend(['--annot', annot_name])
    
    # Optional arguments
    if ctab_file:
        cmd.extend(['--ctab', ctab_file])

def freesurfer_annotation2label(
    subject_dir: str,
    subject_id: str,
    hemi: Optional[str] = None,
    outdir: Optional[str] = None,
    annotation: str = 'aparc',
    labelbase: Optional[str] = None,
    label: Optional[int] = None,
    seg: Optional[str] = None,
    segbase: Optional[int] = None,
    ctab: Optional[str] = None,
    border: Optional[str] = None,
    border_annot: Optional[str] = None,
    surface: str = 'white',
    stat: Optional[str] = None,
    lobes: Optional[str] = None,
    lobes_strict: Optional[str] = None,
    lobes_strict_phcg: Optional[str] = None,
    freesurfer_home: Optional[str] = None
) -> Dict[str, sp.CompletedProcess]:
    """
    Runs freesurfer mri_annotation2label command to convert an annotation file to label files.
    
    Args:
        subject_dir: Path to freesurfer subjects directory
        subject_id: Freesurfer subject ID
        hemi: Hemisphere ('lh', 'rh', or None for both)
        outdir: Output directory for label files
        annotation: Annotation file base name (default: 'aparc')
        labelbase: Base name for label output files
        label: Extract only a single label with this index
        seg: Output segmentation volume file
        segbase: Add this base to annotation number for seg value
        ctab: Color table like FreeSurferColorLUT.txt
        border: Output binary overlay of parcellation borders
        border_annot: Custom location for border annotation
        surface: Surface to use (default: 'white')
        stat: Surface overlay file to use for stats
        lobes: Create annotation based on cortical lobes
        lobes_strict: Create annotation with stricter lobe definition
        lobes_strict_phcg: Create annotation with PHCG lobe definition
        freesurfer_home: FreeSurfer installation directory
        
    Returns:
        Dictionary mapping hemisphere to subprocess.CompletedProcess object
    """
    # Check path objects
    subject_dir = Path(subject_dir)
    subject_path = subject_dir / subject_id
    assert subject_path.exists(), f"Subject not found at {subject_path.absolute()}"
    
    # Set default hemispheres if not specified
    hemispheres = ['lh', 'rh'] if hemi is None else [hemi]
    
    # Find FreeSurfer home
    if freesurfer_home is None:
        freesurfer_home = os.environ.get('FREESURFER_HOME')
        if not freesurfer_home:
            # Common FreeSurfer installation paths
            possible_paths = [
                '/usr/local/freesurfer',
                '/opt/freesurfer',
                '/home/freesurfer',
                '/usr/share/freesurfer'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    freesurfer_home = path
                    break
            if not freesurfer_home:
                raise ValueError("FREESURFER_HOME not found. Please set it manually.")
    
    # Set up environment
    env = os.environ.copy()
    env['SUBJECTS_DIR'] = str(subject_dir.absolute())
    env['FREESURFER_HOME'] = freesurfer_home
    
    # FreeSurfer paths
    fs_bin = os.path.join(freesurfer_home, 'bin')
    fs_lib = os.path.join(freesurfer_home, 'lib')
    
    # Update PATH and LD_LIBRARY_PATH
    env['PATH'] = f"{fs_bin}:{env.get('PATH', '')}"
    if 'LD_LIBRARY_PATH' in env:
        env['LD_LIBRARY_PATH'] = f"{fs_lib}:{env['LD_LIBRARY_PATH']}"
    else:
        env['LD_LIBRARY_PATH'] = fs_lib
    
    # Check if annotation files exist for each hemisphere
    for h in hemispheres:
        annot_path = subject_path / "label" / f"{h}.{annotation}.annot"
        assert annot_path.exists(), f"Annotation not found at {annot_path.absolute()}"
    
    # Set default output directory if not specified
    if outdir is None:
        outdir = subject_path / "label"
    else:
        outdir = Path(outdir)
        os.makedirs(outdir, exist_ok=True)
    
    results = {}
    
    # Run command for each hemisphere
    for h in hemispheres:
        # Build command
        cmd = ['mri_annotation2label',
               f'--subject {subject_id}',
               f'--hemi {h}',
               f'--annotation {annotation}']
        
        # Add optional arguments
        if outdir:
            cmd.append(f'--outdir {outdir}')
        if labelbase:
            cmd.append(f'--labelbase {labelbase}')
        if label is not None:
            cmd.append(f'--label {label}')
        if seg:
            cmd.append(f'--seg {seg}')
        if segbase is not None:
            cmd.append(f'--segbase {segbase}')
        if ctab:
            cmd.append(f'--ctab {ctab}')
        if border:
            cmd.append(f'--border {border}')
        if border_annot:
            cmd.append(f'--border-annot {border_annot}')
        if surface:
            cmd.append(f'--surface {surface}')
        if stat:
            cmd.append(f'--stat {stat}')
        if lobes:
            cmd.append(f'--lobes {lobes}')
        if lobes_strict:
            cmd.append(f'--lobesStrict {lobes_strict}')
        if lobes_strict_phcg:
            cmd.append(f'--lobesStrictPHCG {lobes_strict_phcg}')
        
        # Log the command
        print(f"Running command for {h}:", " ".join(cmd))
        
        # Execute command with better error handling
        try:
            result = sp.run(
                " ".join(cmd),
                shell=True,
                check=True,
                text=True,
                capture_output=True,
                env=env
            )
            print(f"Command for {h} succeeded")
            print(f"Output: {result.stdout}")
            results[h] = result
            
        except sp.CalledProcessError as e:
            print(f"Command for {h} failed with return code {e.returncode}")
            print(f"Error output: {e.stderr}")
            print(f"Standard output: {e.stdout}")
            results[h] = e
    
    return results

def freesurfer_mris_anatomical_stats(
    subject_name: str,
    hemisphere: str,
    subjects_dir: str,
    surface_name: Optional[str] = None,
    thickness_range: Optional[tuple[float, float]] = None,
    label_file: Optional[Union[str, Path]] = None,
    thickness_file: Optional[Union[str, Path]] = None,
    annotation_file: Optional[Union[str, Path]] = None,
    tabular_Returns: bool = False,
    table_file: Optional[Union[str, Path]] = None,
    log_file: Optional[Union[str, Path]] = None,
    smooth_iterations: Optional[int] = None,
    color_table_file: Optional[Union[str, Path]] = None,
    no_global: bool = True,  # Set to True by default to avoid permission issues
    th3: bool = False,
    freesurfer_home: Optional[str] = None
) -> sp.CompletedProcess:
    """
    Run mris_anatomical_stats command with specified parameters.
    """
    # Get FreeSurfer home
    if freesurfer_home is None:
        freesurfer_home = os.environ.get('FREESURFER_HOME')
        if not freesurfer_home:
            # Common FreeSurfer installation paths on Linux
            possible_paths = [
                '/usr/local/freesurfer',
                '/opt/freesurfer',
                '/home/freesurfer',
                '/usr/share/freesurfer'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    freesurfer_home = path
                    break
                    
    if not freesurfer_home:
        raise ValueError("FREESURFER_HOME not found. Please set it manually.")
        
    # Ensure subjects_dir is a string
    if not isinstance(subjects_dir, str):
        subjects_dir = str(subjects_dir)
    
    # Set up environment
    env = os.environ.copy()
    env['SUBJECTS_DIR'] = subjects_dir
    env['FREESURFER_HOME'] = freesurfer_home
    
    # Linux-specific environment variables
    if 'DISPLAY' not in env:
        env['DISPLAY'] = ''  # Needed for X11 forwarding on headless servers
    
    # Set up other required FreeSurfer variables
    fs_bin = os.path.join(freesurfer_home, 'bin')
    fs_lib = os.path.join(freesurfer_home, 'lib')
    
    # Update PATH
    env['PATH'] = f"{fs_bin}:{env.get('PATH', '')}"
    
    # Update LD_LIBRARY_PATH for Linux
    if 'LD_LIBRARY_PATH' in env:
        env['LD_LIBRARY_PATH'] = f"{fs_lib}:{env['LD_LIBRARY_PATH']}"
    else:
        env['LD_LIBRARY_PATH'] = fs_lib
    
    # Check if the command exists
    mris_cmd = os.path.join(fs_bin, 'mris_anatomical_stats')
    if os.path.exists(mris_cmd):
        cmd = [mris_cmd]
    else:
        # Fall back to system PATH
        cmd = ['mris_anatomical_stats']
    
    # Print debug info
    print(f"Using FREESURFER_HOME: {env['FREESURFER_HOME']}")
    print(f"Using SUBJECTS_DIR: {env['SUBJECTS_DIR']}")
    print(f"Command path: {cmd[0]}")
    
    # [Rest of your existing code for building command arguments]
    
    # Add -noglobal flag by default to avoid permission issues
    if no_global:
        cmd.append('-noglobal')
    
    # Add required positional arguments
    cmd.extend([subject_name, hemisphere])
    
    # Add optional positional argument
    if surface_name:
        cmd.append(surface_name)
    
    # Run with better error handling
    try:
        result = sp.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            env=env
        )
        return result
    except sp.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        print(f"Standard output: {e.stdout}")
        raise

def freesurfer_label2vol(
    output_file: str,
    temp_vol: str,
    label_files: Optional[List[str]] = None,
    annot_file: Optional[str] = None,
    seg_file: Optional[str] = None,
    reg_mat_file: Optional[str] = None,
    fill_thresh: Optional[float] = None,
    proj_type: Optional[str] = None,
    proj_start: Optional[float] = None,
    proj_stop: Optional[float] = None,
    proj_delta: Optional[float] = None,
    subject_id: Optional[str] = None,
    hemi: Optional[str] = None,
    identity: bool = False,
    subjects_dir: Optional[str] = None,
    freesurfer_home: Optional[str] = None,
    debug: bool = False
) -> sp.CompletedProcess:
    """
    Converts a label or set of labels into a volume.
    
    Args:
        output_file: Output volume file (any mri_convert format accepted)
        temp_vol: Template volume - output will have same size and geometry
        label_files: List of label files (mutually exclusive with annot_file and seg_file)
        annot_file: Annotation file (mutually exclusive with label_files and seg_file)
        seg_file: Segmentation file (mutually exclusive with label_files and annot_file)
        reg_mat_file: tkregister-style registration matrix file
        fill_thresh: Relative threshold for voxel membership
        proj_type: Projection type ('abs' or 'frac') for surface projection
        proj_start: Start value for projection
        proj_stop: Stop value for projection
        proj_delta: Step size for projection
        subject_id: Subject ID (required when using --proj)
        hemi: Hemisphere ('lh' or 'rh', required when using --proj)
        identity: Use identity matrix as registration
        subjects_dir: FreeSurfer subjects directory
        freesurfer_home: FreeSurfer installation directory
        debug: Print debug information
        
    Returns:
        CompletedProcess object with command results
    """
    # Input validation
    input_count = sum(1 for x in [label_files, annot_file, seg_file] if x is not None)
    if input_count == 0:
        raise ValueError("One of label_files, annot_file, or seg_file must be specified")
    if input_count > 1:
        raise ValueError("Only one of label_files, annot_file, or seg_file can be specified")
    
    # Projection parameter validation
    if any([proj_type, proj_start is not None, proj_stop is not None, proj_delta is not None]):
        if not all([proj_type, proj_start is not None, proj_stop is not None, proj_delta is not None]):
            raise ValueError("All projection parameters (type, start, stop, delta) must be specified when using projection")
        
        if proj_type not in ['abs', 'frac']:
            raise ValueError(f"Projection type must be 'abs' or 'frac', got {proj_type}")
        
        if not subject_id:
            raise ValueError("Subject ID is required when using projection")
        
        if not hemi:
            raise ValueError("Hemisphere is required when using projection")
        
        if hemi not in ['lh', 'rh']:
            raise ValueError(f"Hemisphere must be 'lh' or 'rh', got {hemi}")
    
    # Find FreeSurfer home
    if freesurfer_home is None:
        freesurfer_home = os.environ.get('FREESURFER_HOME')
        if not freesurfer_home:
            # Common FreeSurfer installation paths
            possible_paths = [
                '/usr/local/freesurfer',
                '/opt/freesurfer',
                '/home/freesurfer',
                '/usr/share/freesurfer'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    freesurfer_home = path
                    break
            if not freesurfer_home:
                raise ValueError("FREESURFER_HOME not found. Please set it manually.")
    
    # Set up environment
    env = os.environ.copy()
    if subjects_dir:
        env['SUBJECTS_DIR'] = str(Path(subjects_dir).absolute())
    elif 'SUBJECTS_DIR' not in env and subject_id:
        raise ValueError("SUBJECTS_DIR must be specified either as an argument or environment variable when using subject_id")
    
    env['FREESURFER_HOME'] = freesurfer_home
    
    # FreeSurfer paths
    fs_bin = os.path.join(freesurfer_home, 'bin')
    fs_lib = os.path.join(freesurfer_home, 'lib')
    
    # Update PATH and LD_LIBRARY_PATH
    env['PATH'] = f"{fs_bin}:{env.get('PATH', '')}"
    if 'LD_LIBRARY_PATH' in env:
        env['LD_LIBRARY_PATH'] = f"{fs_lib}:{env['LD_LIBRARY_PATH']}"
    else:
        env['LD_LIBRARY_PATH'] = fs_lib
    
    # Print debug info
    if debug:
        print(f"Using FREESURFER_HOME: {env['FREESURFER_HOME']}")
        print(f"Using SUBJECTS_DIR: {env.get('SUBJECTS_DIR', 'Not set')}")
    
    # Build command
    cmd = ['mri_label2vol']
    
    # Required arguments
    cmd.extend(['--temp', temp_vol])
    cmd.extend(['--o', output_file])
    
    # Input specification (label, annot, or seg)
    if label_files:
        for label in label_files:
            cmd.extend(['--label', label])
    elif annot_file:
        cmd.extend(['--annot', annot_file])
    elif seg_file:
        cmd.extend(['--seg', seg_file])
    
    # Optional arguments
    if reg_mat_file:
        cmd.extend(['--reg', reg_mat_file])
    
    if fill_thresh is not None:
        cmd.extend(['--fillthresh', str(fill_thresh)])
    
    if all([proj_type, proj_start is not None, proj_stop is not None, proj_delta is not None]):
        cmd.extend(['--proj', proj_type, str(proj_start), str(proj_stop), str(proj_delta)])
        cmd.extend(['--subject', subject_id])
        cmd.extend(['--hemi', hemi])
    
    if identity:
        cmd.append('--identity')
    
    if subjects_dir:
        cmd.extend(['--sd', subjects_dir])
    
    # Log the command
    cmd_str = ' '.join(cmd)
    if debug:
        print(f"Running command: {cmd_str}")
    
    # Execute command with error handling
    try:
        result = sp.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            env=env
        )
        if debug:
            print("Command succeeded")
            print(f"Output: {result.stdout}")
        return result
        
    except sp.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        print(f"Standard output: {e.stdout}")
        raise
    


def get_subjects_list(subjects_list: str, subjects_dir: str) -> list:
    '''
    Takes a txt subject list and returns a list of subject directory filepaths
    
    Args:
    subjects_list: str = filepath to .txt subject list
    subjects_dir: str = filepath to subjects directory

    Returns:
    subjects_filepaths: list: list of subject filepaths as strings
    '''
    
    with open(subjects_list, 'r', encoding="utf-8") as list_file:
        subject_names = [line.rstrip() for line in list_file]

    subject_filepaths = []
   
    for subject in subject_names:
        subject_filepath = os.path.join(subjects_dir, subject)
        
        assert os.path.exists(subject_filepath), f"{subject} does not exist within SUBJECTS_DIR {subjects_dir}"

        subject_filepaths.append(subject_filepath)
    
    return subject_filepaths
    


def sort_subjects_and_sulci(subject_filepaths: list, sulci_list: list) -> dict:
    '''
    Sorts subject hemispheres into groups based on which sulci are present in each hemisphere

    Args:
    subject_filepath : list - output of get_subjects_list, a list of all full paths to subjects

    sulci_list : list - all possible sulci

    Returns:
    subject_sulci_dict : dict - {subject_id : [[lh_sulci_present, rh_sulci_present]]}
    '''
    
    subject_sulci_dict = {}

    ### for subjects, check which paths exist and which dont

    for sub_path in subject_filepaths:
        for hemi in ['lh', 'rh']:
            subject_path = Path(sub_path)
            subject_id = subject_path.name
            assert subject_path.exists(), f"{subject_id} does not exist at {subject_path}"
            
            subject_label_paths = get_sulci_filepaths(sub_path, sulci_list, hemi)
            existing_subject_labels_by_hemi = []

            for i, label in enumerate(sulci_list):
                if subject_label_paths[i].exists():
                    #print(f"{subject_id} has the {hemi} {label} label")
                    existing_subject_labels_by_hemi.append(label)
                else:
                    #print(f"{subject_id} does not have the {hemi} {label} label")
                    pass
            
    
    ##  add to dictionary key fo subject_id based on label existenc
            subject_sulci_dict[f"{hemi}_{subject_id}"] = existing_subject_labels_by_hemi

    return subject_sulci_dict



def get_sulci_filepaths(subject_filepath: str, sulci_list: list, hemi: str) -> list:
    '''
    Takes a subject path, list of sulci, and hemisphere and returns list of sulci label paths
    '''      
    subject_filepath = Path(subject_filepath)
    assert subject_filepath.exists(), f"The subject file path does not exist: {subject_filepath}"

    label_paths = [subject_filepath / 'label' / f'{hemi}.{label}.label' for label in sulci_list]

    return label_paths



def create_freesurfer_ctab(ctab_name: str, label_list: list, outdir: str, palette: dict = None ):
    '''
    Creates a color table file for label2annot 
    
    Args:
    ctab_name : str - desired name of color table
    label_list : list - list of strings containing all desired labels
    outdir : str - desired output directory
    pallete : list - custom colors - dict labels and rgb colors as strings, with rgb values separated by tab - i.e. ['MFS' : 'int<tab>int<tab>int', ...]
    '''
    
    outdir_path = Path(outdir)
    assert outdir_path.exists(), f"{outdir.resolve()} does not exist"

    ctab_path = f"{outdir}/{ctab_name}.ctab"
    date = datetime.datetime.now()

    if palette is None:        
        palette = {f"{label}" : f"{randint(low=1, high=248)} {randint(low=1, high=248)} {randint(low=1, high=248)}"  for label in label_list}
    else:
        pass

    with open(ctab_path, 'w', encoding="utf-8") as file:
        file.write(f'#$Id: {ctab_path}, v 1.38.2.1 {date.strftime("%y/%m/%d")} {date.hour}:{date.minute}:{date.second} CNL Exp $ \n')
        file.write("No. Label Name:                R   G   B   A\n")
        file.write("0  Unknown         0   0   0   0\n")
        for i, label_name in enumerate(label_list):
            file.write(f"{i + 1}    {label_name}                {palette[label_name]}  0\n")

    

def create_ctabs_from_dict(project_colortable_dir: str, sulci_list: list, json_file: str, project_name : str, palette: dict = None):
    ''' 
    Takes a dictionary of subjects and present sulci,
    creates a colortable file for each unique combination of sulci

    Args:
    project_colortable_dir : str - filepath to project colortable directory
    json_file : str - filepath to json file containing subject sulci dictionary
    sulci_list : list - list of all possible sulci
    palette : dict - custom colors - dict labels and rgb colors as strings, with rgb values separated by tab - i.e. ['MFS' : 'int<tab>int<tab>int', ...]
    project_name : str - unique identifier for project 

    Returns:
    ctab files for each unique combination of sulci
    '''
    print(json_file)
    with open(json_file, 'r', encoding="utf-8") as file:
        sulci_dict = json.load(file)

    # get all sulci in dictionary
    all_sulci_in_dict = list(sulci_dict.values())
    
    # get unique combinations of sulci 
    unique_sulci_lists = [list(sulc_list) for sulc_list in set(tuple(sulc_list) for sulc_list in all_sulci_in_dict)]
    
    if palette is None:        
        palette = {f"{label}" : f"{randint(low=1, high=248)} {randint(low=1, high=248)} {randint(low=1, high=248)}"  for label in sulci_list}
    else:
        print(palette.keys())
        print(sulci_list)
        
        assert len(palette.keys()) == len(sulci_list), "Palette length does not match label list length"

    # store unique comnbinations of sulci in dictionary, with key by indexed combination number
    ctab_file_dict = {}

    # this is done to avoid file length limitations when having all sulci in filename (linux=255 bytes)
    # match subject hemi entry to value in the ctab_file_dict

    for i, unique_sulci_list in enumerate(unique_sulci_lists):
        num_sulci = len(unique_sulci_list)
        ctab_name = f'{project_name}_ctab_{i}_{num_sulci}_sulci'
        ctab_file_dict[ctab_name] = unique_sulci_list
    
    dict_to_json(dictionary = ctab_file_dict, outdir = project_colortable_dir, project_name = f"{project_name}_ctab_files")
    
    # Get custom palette for each sulcus
    for key, value in ctab_file_dict.items():
        custom_palette = dict((val, palette[val]) for val in value)
        create_freesurfer_ctab(ctab_name=key, label_list=value,
                            outdir=project_colortable_dir, palette=custom_palette)

        
        

def dict_to_json(dictionary: dict, outdir: str, project_name: str):
    '''
    Takes a dictionary and saves as a JSON

    Args:
    dictionary : dict - dictionary of {hemi_subject_id, [sulci_list]} created by sort_subjects_and_sulci()
    outdir : str - write directory for json of colortables
            NOTE: should be written to project directory for colortables
    project_name : str - the name of the project to be the name of the .json i.e. voorhies_natcom_2021.json

    Returns:
    .json file of dictionary
    '''
    print(outdir)
    assert os.path.exists(outdir), f"{outdir} does not exist"
    
    save_file = os.path.join(outdir, f"{project_name}.json")

    with open(save_file, 'w', encoding="utf-8") as file:
        json.dump(dictionary, file, indent=4)


def rename_labels(subjects_dir: str, subjects_list: str, sulci_dict: dict, by_copy: bool = True):
    '''
    Renames labels in a given hemisphere for all subjects in a given subjects list

    Args:
    subjects_dir : str - filepath to subjects directory
    subjects_list : str - filepath to subjects list
    sulci_list : dict - dict of sulci,{old_name: new_name}
    by_copy : bool - if True, copies files by cp (keeps original file) ; if False, renames files by mv (deletes original file)

    Returns:
    Renamed label files
    
    '''
    assert os.path.exists(subjects_dir), f"{subjects_dir} does not exist"
    assert os.path.exists(subjects_list), f"{subjects_list} does not exist"
    
    subject_filepaths = get_subjects_list(subjects_list, subjects_dir)
    
    if by_copy is True:
        # Copies files by cp (keeps original file)
        for subject_path in subject_filepaths:
    
            assert os.path.exists(subject_path), f"The subject does not exist at {subject_path}"

            for hemi in ['lh', 'rh']:
                for sulcus in sulci_dict.items():
                    cmd = f"cp {subject_path}/label/{hemi}.{sulcus[0]}.label {subject_path}/label/{hemi}.{sulcus[1]}.label"
                    print(f"Executing: {cmd}")
                    run_cmd = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)

                    out, err = run_cmd.communicate()

                    if run_cmd.returncode == 0:
                        pass
                    else:
                        print(f"out: {out}")
                        print(f"err: {err}")
                        print(f'Be sure that {hemi}.{sulcus[0]}.label exists in {subject_path}')
                    

    else:
        # Renames files by mv (removes original file)
        for subject_path in subject_filepaths:
    
            assert os.path.exists(subject_path), f"The subject does not exist at {subject_path}"

            for hemi in ['lh', 'rh']:
                for sulcus in sulci_dict.items():
                    
                    cmd = f"mv {subject_path}/label/{hemi}.{sulcus[0]}.label {subject_path}/label/{hemi}.{sulcus[1]}.label"
                    print(f"Executing: {cmd}")
                    run_cmd = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)

                    out, err = run_cmd.communicate()

                    if run_cmd.returncode == 0:
                        pass
                    else:
                        print(f"out: {out}")
                        print(f"err: {err}")
                        print(f'Be sure that {hemi}.{sulcus[0]}.label exists in {subject_path}')


def create_tar_from_subject_list(project_dir: str, tarfile_name: str, subject_list: str, subjects_dir: str): 
    """
    Creates a compressed .tar.gz file from a list of subjects recursively. 
        NOTE: This will add ALL files located in freesurfer subject directory
    Args:
    project_dir : str - filepath to project directory where tar will be written
    tarfile_name : str - name for tar archive
    subject_list : str - filepath to .txt list of subjects
    subjects_dir : str - filepath freesurfer subjects directory

    
    """
    # Get subject list
    subject_list = get_subjects_list(subjects_dir=subjects_dir, subjects_list=subject_list)
    
    # check suffix and remove
    if tarfile_name[-7:] == '.tar.gz':
        tarfile_name = tarfile_name[:-7]
    
    assert os.path.exists(project_dir), "{project_dir} does not exist"

    
    # Check if tar exists
    try:
        with tarfile.open(f"{project_dir}/{tarfile_name}.tar.gz", mode='x:gz') as tar:
            print(f'Creating {tarfile_name} \n')
            for subject_dir in subject_list:
                tar.add(subject_dir, recursive=True)
            print('tarfile created.')
    except FileExistsError:
        # if tar exists, confirm user wants to add new subjects to tar
        print(f'\n {tarfile_name}.tar.gz already exists. \n')

        add_to_tar = input('Do you want to add the subjects to this existing tarfile? [y/n] ').lower()
        if add_to_tar in ['y', 'yes']:
         print('\nAdding\n')
        
         with tarfile.open(f"{project_dir}{tarfile_name}.tar.gz", mode='w:gz') as tar:
            for subject_dir in subject_list:
                tar.add(subject_dir, recursive=True)
        else: 
            print(f'\nSubjects not added to {tarfile_name}.\n')


def create_tar_for_file_from_subject_list(project_dir: str, tarfile_name: str, subject_list: str, subjects_dir: str, filepath_from_subject_dir : str): 
    """
    Creates a compressed .tar.gz file from a list of subjects recursively. 
        NOTE: This will add ALL files located in freesurfer subject directory
    Args:
    project_dir : str - filepath to project directory where tar will be written
    tarfile_name : str - name for tar archive
    subject_list : str - filepath to .txt list of subjects
    subjects_dir : str - filepath freesurfer subjects directory


    """
    # Get subject list
    subject_list = get_subjects_list(subjects_dir=subjects_dir, subjects_list=subject_list)
    
    # check suffix and remove
    if tarfile_name[-7:] == '.tar.gz':
        tarfile_name = tarfile_name[:-7]
    
    assert os.path.exists(project_dir), "{project_dir} does not exist"

    
    # Check if tar exists
    try:
        with tarfile.open(f"{project_dir}/{tarfile_name}.tar.gz", mode='x:gz') as tar:
            print(f'Creating {tarfile_name} \n')
            for subject_dir in subject_list:
                
                tar.add(f"{subject_dir}/{filepath_from_subject_dir}")
            print('tarfile created.')
    except FileExistsError:
        # if tar exists, confirm user wants to add new subjects to tar
        print(f'\n {tarfile_name}.tar.gz already exists. \n')

        add_to_tar = input('Do you want to add the subjects to this existing tarfile? [y/n] ').lower()
        if add_to_tar in ['y', 'yes']:
         print('\nAdding\n')
        
         with tarfile.open(f"{project_dir}{tarfile_name}.tar.gz", mode='w:gz') as tar:
            for subject_dir in subject_list:
                tar.add(f"{subject_dir}/{filepath_from_subject_dir}")
        else: 
            print(f'\nSubjects not added to {tarfile_name}.\n')


def read_label(label_name: Union[str, Path], include_stat: bool = False) -> tuple:
    """
    Reads a freesurfer-style .label file (5 columns)
    
    Parameters
    ----------
    label_name: str or Path - name of label file to be read
    
    Returns 
    -------
    vertices: index of the vertex in the label np.array [n_vertices] 
    RAS_coords: columns are the X,Y,Z RAS coords associated with vertex number in the label, np.array [n_vertices, 3] 
    incude_stat: bool - if True, includes the statistic / value of the vertex in the label (fifth column in .label file)
    
    """
    
    # read label file, excluding first two lines of descriptor 
    df_label = pd.read_csv(label_name,skiprows=[0,1],header=None,names=['vertex','x_ras','y_ras','z_ras','stat'],delimiter='\s+')
    
    vertices = np.array(df_label.vertex) 
    RAS_coords = np.empty(shape = (vertices.shape[0], 3))
    RAS_coords[:,0] = df_label.x_ras
    RAS_coords[:,1] = df_label.y_ras
    RAS_coords[:,2] = df_label.z_ras

    if include_stat:
        stat = df_label.stat
        return vertices, RAS_coords, stat
    
    return vertices, RAS_coords

def write_label( label_name: Union[str, Path], label_indexes: np.array, label_RAS: np.array, hemi: str, subject_dir: Union[str, Path], surface_type: str, overwrite: bool = False, **kwargs):
    """
    Write freesurfer label file from label indexes and RAS coordinates

    Args:
    label_name: str - name of label to be written
    label_indexes: np.array - numpy array of label indexes
    label_RAS: np.array - numpy array of label RAS coordinates
    hemi: str - hemisphere
    subject_dir: str or Path - subject directory
    surface_type: str - surface type for label
    overwrite: bool - overwrite existing label file
    custom_label_name: str - custom label name for label file (optional) - shhould be complete string literal of label name in subject label file i.e. 'custom.label.name.label'
    custom_label_dir: str - custom label directory for label file (optional) - should be complete string literal of label directory in subject label file
    
    """
    
    if isinstance(subject_dir, str):
        subject_dir = Path(subject_dir)
    
    ## Check for custom label directory
    if 'custom_label_dir' in kwargs:
        if isinstance(kwargs['custom_label_dir'], str):
            label_dir = Path(kwargs['custom_label_dir'])
        else:
            label_dir = kwargs['custom_label_dir']
    else:
        label_dir = subject_dir / 'label'

    ## Check for custom label name
    if 'custom_label_name' in kwargs:
        label_name = kwargs['custom_label_name']
    else:
        label_name = f"{hemi}.{label_name}.label"
    
    ## Create full filename
    label_filename = label_dir / label_name
    
    if not overwrite:
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

def get_sulcus(label_index: np.array, label_RAS: np.array, curv: np.array, curv_threshold: int = 0):
    """ 
    Returns all label indices and RAS coordinates for sulcus within freesurfer label

    Args:
    _____
    label_index: np.array - numpy array of label indexes from src.read_label()
    label_RAS: np.array - numpy array of label RAS vertices from src.read_label()
    curv: np.array - numpy array of curvature values from nb.freesurfer.read_morph_data()
    curv_threshold: int - value for thresholding curvature value

    Returns:
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


def get_gyrus(label_index: np.array, label_RAS: np.array, curv: np.array, curv_threshold: int = 0):
    """ 
    Returns all label indices and RAS coordinates for gyrus within freesurfer label

    Args:
    _____
    label_index: np.array - numpy array of label indexes from src.read_label()
    label_RAS: np.array - numpy array of label RAS vertices from src.read_label()
    curv: np.array - numpy array of curvature values from nb.freesurfer.read_morph_data()
    curv_threshold: int - value for thresholding curvature value

    Returns:
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


def mris_anatomical_stats2DataFrame_row(subject: str, label_name: str, hemi: str, data_dir: Union[str, Path]) -> pd.DataFrame:
    """ 
    Takes a subject list and the location of a stats.txt file outputted by mris_anatomical_stats ->> converts it to a dataframe

    Args:
    subject: str - subject ID
    label_name: str - name of the label to be included in the dataframe
    hemi: str - hemisphere to be included in the dataframe (must be 'lh', 'rh')
    data_dir: str or Path - directory where the stats.txt file is located

    Returns:
    pd.DataFrame


    """

    assert hemi in ['lh', 'rh'], "hemi must be 'lh' or 'rh'"

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    txt_path = data_dir / f"{hemi}.{label_name}.label.stats.txt"
    assert txt_path.exists(), f'the file {txt_path} does not exist.'

    all_stats_df = pd.DataFrame(columns=['sub', 'hemi', 'label', 'num_vertices', 'surface_area_mm^2', 'gray_matter_volume_mm^3', 'avg_cortical_thickness', 'avg_cortical_thickness_std', 'rectified_mean_curvature', 'rectified_gaussian_curvature', 'folding_index', 'intrinsic_curvature'])

    with open(txt_path, 'r') as fp:
        new_surf = fp.readlines()
    
    row_stats = new_surf[-1]
    row_stats = row_stats.split(' ')
    row_stats = [i for i in row_stats if i != '']
    label_name = row_stats[-1][:-1].split('.')[1]

    num_rows =  [row_stats[0], row_stats[1], row_stats[2], row_stats[3], row_stats[4], row_stats[5], row_stats[6], row_stats[7], row_stats[8]]
    num_rows = [float(i) for i in num_rows]
    all_stats_row = [subject, hemi, label_name, num_rows[0], num_rows[1], num_rows[2], num_rows[3], num_rows[4], num_rows[5], num_rows[6], num_rows[7], num_rows[8]]
    
    all_stats_df.loc[len(all_stats_df)] = all_stats_row

    return all_stats_df


def subject_label_stats2DataFrame(subjects_dir: Union[str, Path], subject_list: list, label_name: Union[str, list], hemi: Union[str, list], data_dir_from_subject_fs_dir:str = 'label', must_exist = True) -> pd.DataFrame:
    """ 
    Takes a subject list, label list, and the location of a stats.txt file outputted by mris_anatomical_stats ->> converts it to a dataframe

    Args:
    subjects_dir: str or Path - FreeSurfer subjects directory
    subject_list: list - list of subjects to be included in the dataframe
    label_name: str or list - name of the label to be included in the dataframe
    hemi: str or list - hemisphere to be included in the dataframe (must be 'lh', 'rh', or 'both')
    data_dir_from_subject_fs_dir: str - directory where the stats.txt file is located relative to the subject's FreeSurfer directory, default is 'label'

    Returns: 
    pd.DataFrame

    """

    assert hemi in ['lh', 'rh', 'both'], "hemi must be 'lh', 'rh', or 'both'"
    if hemi == 'both':
        hemi_list = ['lh', 'rh']
    else:
        hemi_list = [hemi]

    if isinstance(label_name, str):
        label_name = [label_name]
        
    if isinstance(subjects_dir, str):
        subjects_dir = Path(subjects_dir)

    all_stats_df = pd.DataFrame(columns=['sub', 'hemi', 'label', 'num_vertices', 'surface_area_mm^2', 'gray_matter_volume_mm^3', 'avg_cortical_thickness', 'avg_cortical_thickness_std', 'rectified_mean_curvature', 'rectified_gaussian_curvature', 'folding_index', 'intrinsic_curvature'])

    
    for sub in subject_list:
        for hemi in hemi_list:
            for label in label_name:
                data_dir = subjects_dir / sub / data_dir_from_subject_fs_dir 
                if must_exist:
                    assert data_dir.exists(), f"{data_dir} does not exist"
                else:
                    try:
                        new_row = mris_anatomical_stats2DataFrame_row(sub, label, hemi, data_dir)
                        print(new_row)
                        all_stats_df = pd.concat([all_stats_df, new_row], axis = 0)
                    except:
                        pass

    return all_stats_df


############################################################################################################
############################################################################################################
# Maximum Probability Map
############################################################################################################
############################################################################################################

## Project labels to fsaverage

    ## Create probabilty maps for each subject, with that subject held out
def create_prob_label(project_id: str, fsaverage_projected_label_dir: str, subject_list_path: str, prob_map_label_dir: str, label_name: str, left_out_subject: str,  hemi: str):
            """ 
            Creates probabilistic label files for a given label, with a subject held out

            Args:
            project_id: str - unique identifier for project (included in final name of probabilistic label)
            fsaverage_projected_label_dir: str - filepath to fsaverage projected labels (resulting from freesurfer_label2label)
            subject_list_path: str - filepath to subject list
            prob_map_label_dir: str - filepath to probabilistic label directory
            label_name: str - name of the label
            left_out_subject: str - subject to be held out
            hemi: str - hemisphere

            Returns:
            probabilistic label files for each subject, with the left out subject held out 
            """
            ## Load subjects and remove left out subject
            subjects = np.genfromtxt(subject_list_path) 

            subjects = [i for i in subjects if i != left_out_subject]

            ## Create empty array for vertices of projected labels
            label_vertices= np.empty(shape=0,dtype=int)
            label_RAS = np.empty(shape=(0,3),dtype=int)

            prob_label_name = f'{hemi}.{project_id}_PROB_{label_name}.label'
            prob_label_dir = prob_map_label_dir + f'/{left_out_subject}/'
            os.makedirs(prob_label_dir, exist_ok=True)

            ## Loop through subjects, load projected labels, and append vertices and RAS coords to arrays
            for sub in subjects:
                # Load projected label
                label_path = fsaverage_projected_label_dir + f'/projected_labels/{label_name}/{sub}.lh.{label_name}.label'
                vertices, RAS = read_label(label_path)
                
                # Append vertices from projected label to array
                label_vertices = np.append(label_vertices, vertices)
                label_RAS = np.append(label_RAS,RAS,axis=0)
                
                # Update unique vertices and counts
                unique_vertices, indices_vertices, counts_vertices=np.unique(label_vertices,return_index=True,return_counts=True)

                # index only the RAS coords for unique vertices
                unique_RAS = label_RAS[indices_vertices,:]

                # get probabilities at each vertex 
                prob_vertices = (counts_vertices)/len(subjects)
                

            # make probabilistic label array for label file
            prob_array = np.zeros(shape=(unique_vertices.shape[0],5),dtype=float)
            prob_array[:,0] = unique_vertices
            prob_array[:,1:4] = unique_RAS
            prob_array[:,-1] = prob_vertices

            # write_label(label_name = label_name, label_indices = unique_vertices, label_RAS = unique_RAS, hemi = hemi, 
            #             custom_label_dir = prob_map_label_dir, custom_label_name = prob_label_name) 
            #     # np.savetxt(prob_label_path, prob_array, fmt='%-2d  %2.3f  %2.3f  %2.3f %1.10f')

            # edit first two lines of label file to match Freesurfer
            f = open(prob_label_path, 'r')
            edit_f = f.read()
            f.close()
            f = open(prob_label_path, 'w')
            f.write('#!ascii label  , from subject fsaverage vox2ras=TkReg\n{}\n'.format(unique_vertices.shape[0]))
            f.write(edit_f)
            f.close()

## Create Maximum Probability Map
def create_MPM(subjects_dir: str, subjects_list: str, fsaverage_space_labels: str, project_id: str):
    """
    """

    prob_maps_vertices = {}
    prob_maps_RAS = {}
    prob_maps_stat = {}

    MPM_vertices = {}
    MPM_RAS = {}
    MPM_stat = {}

    print('Creating Left Hemisphere MPMs \n\n\n')

    # loop through labels, load prob map and make empty values in MPM dicts
    for i, middle_frontal_label in enumerate(middle_frontal_label_names):
        try:
            #load the prob mpa for the given label
            vertices, RAS, stat = read_label(fsaverage_space_labels + 'prob_maps/{}/lh.{}_PROB_{}.label'.format(left_out_sub, project_id, middle_frontal_label))
            prob_maps_vertices[middle_frontal_label] = vertices
            prob_maps_RAS[middle_frontal_label] = RAS
            prob_maps_stat[middle_frontal_label] = stat

            MPM_vertices[middle_frontal_label] = np.empty(0)
            MPM_RAS[middle_frontal_label] =  np.empty(shape=(0,3))
            MPM_stat[middle_frontal_label] = np.empty(0)
        except Exception:
            pass
            #load lh cortex vertices
    vertices_lh, RAS_lh = read_label('/home/weiner/data/fsaverage/label/lh.cortex.label')
        

    # loop through lh cortex vertices
    for vtx in vertices_lh:

        labels_with_vtx = np.empty(0)
        vertices_with_vtx = np.empty(0)
        RAS_with_vtx = np.empty(shape=(0,3))
        stat_with_vtx = np.empty(0)

        for label_prob, vertices_prob in prob_maps_vertices.items():
            # if vtx from cortex is in vertices of prob map, add name to list of labels with vtx, add stat value
            match_idx = np.where(vertices_prob == vtx)
            # if vertex is in probability map
            if match_idx[0].shape[0] > 0:
                # get vertex, RAS, and stat values for the given vertex that is in prob map
                vertices_prob_idx = match_idx[0][0]
                labels_with_vtx = np.append(labels_with_vtx, label_prob)
                vertices_with_vtx = np.append(vertices_with_vtx, vertices_prob[vertices_prob_idx])
                RAS_with_vtx = np.concatenate((RAS_with_vtx, np.reshape(prob_maps_RAS[label_prob][vertices_prob_idx,:],(1,3))),axis=0)
                stat_with_vtx = np.append(stat_with_vtx, prob_maps_stat[label_prob][vertices_prob_idx])

        # if vertex was in at least one prob map, get the max value and add to MPM file
            # also required to have a probability of 0.33 or higher
        if (labels_with_vtx.shape[0] > 0):

            if (np.max(stat_with_vtx) > 1/3):

                max_idx = np.argmax(stat_with_vtx)

                label_max = labels_with_vtx[max_idx]
                RAS_max = RAS_with_vtx[max_idx,:]
                stat_max = stat_with_vtx[max_idx]

                MPM_vertices[label_max] = np.append(MPM_vertices[label_max], vtx)
                MPM_RAS[label_max] = np.concatenate((MPM_RAS[label_max], np.reshape(RAS_max,(1,3))),axis=0)
                MPM_stat[label_max] = np.append(MPM_stat[label_max], stat_max)

    # save out each entry MPM as a separate label file in Freesurfer

    for i, middle_frontal_label in enumerate(middle_frontal_label_names):

        MPM_path = fsaverage_space_labels + 'prob_maps/{}/lh.{}_PROB_MPM_{}.label'.format(left_out_sub, project_id, middle_frontal_label)
        try:
            # make probabilistic label array for albel file
            prob_array = np.zeros(shape=(MPM_vertices[middle_frontal_label].shape[0],5),dtype=float)
            prob_array[:,0] = MPM_vertices[middle_frontal_label]
            prob_array[:,1:4] = MPM_RAS[middle_frontal_label]
            prob_array[:,-1] = MPM_stat[middle_frontal_label]
            np.savetxt(MPM_path, prob_array, fmt='%-2d  %2.3f  %2.3f  %2.3f %1.10f')

            # edit first two lines of label file to match Freesurfer
            f = open(MPM_path, 'r')
            edit_f = f.read()
            f.close()
            f = open(MPM_path, 'w')
            f.write('#!ascii label  , from subject fsaverage vox2ras=TkReg\n{}\n'.format(MPM_vertices[middle_frontal_label].shape[0]))
            f.write(edit_f)
            f.close()
        except Exception:
            pass
    print('Left Hemisphere PROB MPMs written for ', left_out_sub,' \n\n\n') 


    # loop through labels, load prob map and make empty values in MPM dicts
    for i, middle_frontal_label in enumerate(middle_frontal_label_names):
        try:
            #load the prob mpa for the given label
            vertices, RAS, stat = read_label_stat(fsaverage_space_labels + 'prob_maps/{}/lh.{}_PROB_{}.label'.format(left_out_sub, project_id, middle_frontal_label))
            prob_maps_vertices[middle_frontal_label] = vertices
            prob_maps_RAS[middle_frontal_label] = RAS
            prob_maps_stat[middle_frontal_label] = stat

            MPM_vertices[middle_frontal_label] = np.empty(0)
            MPM_RAS[middle_frontal_label] =  np.empty(shape=(0,3))
            MPM_stat[middle_frontal_label] = np.empty(0)
        except Exception:
            pass
            #load lh cortex vertices
    vertices_lh, RAS_lh = read_label('/home/weiner/data/fsaverage/label/lh.cortex.label')
        

    # loop through lh cortex vertices
    for vtx in vertices_lh:

        labels_with_vtx = np.empty(0)
        vertices_with_vtx = np.empty(0)
        RAS_with_vtx = np.empty(shape=(0,3))
        stat_with_vtx = np.empty(0)

        for label_prob, vertices_prob in prob_maps_vertices.items():
            # if vtx from cortex is in vertices of prob map, add name to list of labels with vtx, add stat value
            match_idx = np.where(vertices_prob == vtx)
            # if vertex is in probability map
            if match_idx[0].shape[0] > 0:
                # get vertex, RAS, and stat values for the given vertex that is in prob map
                vertices_prob_idx = match_idx[0][0]
                labels_with_vtx = np.append(labels_with_vtx, label_prob)
                vertices_with_vtx = np.append(vertices_with_vtx, vertices_prob[vertices_prob_idx])
                RAS_with_vtx = np.concatenate((RAS_with_vtx, np.reshape(prob_maps_RAS[label_prob][vertices_prob_idx,:],(1,3))),axis=0)
                stat_with_vtx = np.append(stat_with_vtx, prob_maps_stat[label_prob][vertices_prob_idx])

        # if vertex was in at least one prob map, get the max value and add to MPM file
            # also required to have a probability of 0.33 or higher
        if (labels_with_vtx.shape[0] > 0):

            if (np.max(stat_with_vtx) > 1/3):

                max_idx = np.argmax(stat_with_vtx)

                label_max = labels_with_vtx[max_idx]
                RAS_max = RAS_with_vtx[max_idx,:]
                stat_max = stat_with_vtx[max_idx]

                MPM_vertices[label_max] = np.append(MPM_vertices[label_max], vtx)
                MPM_RAS[label_max] = np.concatenate((MPM_RAS[label_max], np.reshape(RAS_max,(1,3))),axis=0)
                MPM_stat[label_max] = np.append(MPM_stat[label_max], stat_max)

    # save out each entry MPM as a separate label file in Freesurfer

    for i, middle_frontal_label in enumerate(middle_frontal_label_names):

        MPM_path = fsaverage_space_labels + 'prob_maps/{}/lh.{}_PROB_MPM_{}.label'.format(left_out_sub, project_id, middle_frontal_label)
        try:
            # make probabilistic label array for albel file
            prob_array = np.zeros(shape=(MPM_vertices[middle_frontal_label].shape[0],5),dtype=float)
            prob_array[:,0] = MPM_vertices[middle_frontal_label]
            prob_array[:,1:4] = MPM_RAS[middle_frontal_label]
            prob_array[:,-1] = MPM_stat[middle_frontal_label]
            np.savetxt(MPM_path, prob_array, fmt='%-2d  %2.3f  %2.3f  %2.3f %1.10f')

            # edit first two lines of label file to match Freesurfer
            f = open(MPM_path, 'r')
            edit_f = f.read()
            f.close()
            f = open(MPM_path, 'w')
            f.write('#!ascii label  , from subject fsaverage vox2ras=TkReg\n{}\n'.format(MPM_vertices[middle_frontal_label].shape[0]))
            f.write(edit_f)
            f.close()
        except Exception:
            pass
    print('Left Hemisphere PROB MPMs written for ', left_out_sub,' \n\n\n') 

    # save out each entry MPM as a separate binary label file in Freesurfer

    for i, middle_frontal_label in enumerate(middle_frontal_label_names):

        MPM_path = fsaverage_space_labels + 'prob_maps/{}/lh.{}_PROB_MPM_binary_{}.label'.format(left_out_sub, project_id ,middle_frontal_label)
        try:
            # make probabilistic label array for albel file
            prob_array = np.zeros(shape=(MPM_vertices[middle_frontal_label].shape[0],5),dtype=float)
            prob_array[:,0] = MPM_vertices[middle_frontal_label]
            prob_array[:,1:4] = MPM_RAS[middle_frontal_label]
            prob_array[:,-1] = 1
            np.savetxt(MPM_path, prob_array, fmt='%-2d  %2.3f  %2.3f  %2.3f %1.10f')

            # edit first two lines of label file to match Freesurfer
            f = open(MPM_path, 'r')
            edit_f = f.read()
            f.close()
            f = open(MPM_path, 'w')
            f.write('#!ascii label  , from subject fsaverage vox2ras=TkReg\n{}\n'.format(MPM_vertices[middle_frontal_label].shape[0]))
            f.write(edit_f)
            f.close()

        except Exception:
            pass
        
    print('Left Hemisphere PROB Binary MPMs written for ', left_out_sub, '\n\n\n')


