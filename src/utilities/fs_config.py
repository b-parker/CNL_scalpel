
import os
from pathlib import Path
import json
import warnings

def get_freesurfer_home():
    """Get FreeSurfer home from environment or config file."""
    # First check environment
    fs_home = os.environ.get('FREESURFER_HOME')
    if fs_home:
        return fs_home
    
    # Then check config file
    config_file = Path('.') / '.fs_config' / 'config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                fs_home = config.get('FREESURFER_HOME')
                if fs_home:
                    # Set for current process
                    os.environ['FREESURFER_HOME'] = fs_home
                    return fs_home
        except Exception:
            pass
    
    warnings.warn("FREESURFER_HOME not found. Run 'python -m your_package.setup_freesurfer' to configure.")
    return None