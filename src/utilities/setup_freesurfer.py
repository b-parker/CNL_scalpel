# setup_freesurfer.py
import os
import sys
from pathlib import Path
import json

def setup_freesurfer_environment():
    print("FreeSurfer Configuration Setup")
    print("==============================")
    
    # Check if FREESURFER_HOME is already set
    current_fs_home = os.environ.get('FREESURFER_HOME')
    
    if current_fs_home:
        print(f"FREESURFER_HOME is currently set to: {current_fs_home}")
        change = input("Would you like to change this? (y/n): ").lower().strip()
        if change != 'y':
            print("Keeping current FreeSurfer configuration.")
            return current_fs_home
    
    # Get FreeSurfer home path from user
    fs_home = input("\nPlease enter the path to your FreeSurfer installation: ").strip()
    
    # Validate the path
    fs_path = Path(fs_home)
    if not fs_path.exists():
        print(f"Warning: {fs_home} does not exist.")
        print("Please verify your FreeSurfer installation.")
        return None
    
    # Check for bert subject
    bert_path = fs_path / "subjects" / "bert"
    if not bert_path.exists():
        print(f"Warning: Sample subject 'bert' not found at {bert_path}")
        print("Some tests may be skipped without this sample data.")
    
    # Create a local project configuration
    config_dir = Path('.') / '.fs_config'
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / 'config.json'
    try:
        with open(config_file, 'w') as f:
            json.dump({'FREESURFER_HOME': fs_home}, f, indent=2)
        print(f"Created configuration file at {config_file}")
    except Exception as e:
        print(f"Error creating configuration file: {e}")
    
    # Provide instructions for environment setup
    print("\nFreeSurfer configuration saved!")
    print(f"FREESURFER_HOME set to: {fs_home}")
    print("\nTo use with your current shell session, run:")
    
    if os.name == 'nt':  # Windows
        print(f'set FREESURFER_HOME={fs_home}')
    else:  # Unix/Linux/MacOS
        print(f'export FREESURFER_HOME={fs_home}')
    
    if 'CONDA_PREFIX' in os.environ:
        print("\nTo set permanently in your conda environment, run:")
        print(f'conda env config vars set FREESURFER_HOME={fs_home}')
        print("Then reactivate your environment with: conda activate <env_name>")
    else:
        print("\nTo set permanently, add to your environment or virtual environment.")
    
    return fs_home

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
                return config.get('FREESURFER_HOME')
        except Exception:
            pass
    
    return None

if __name__ == "__main__":
    setup_freesurfer_environment()