import pytest
import os
from pathlib import Path
from src.utilities.fs_config import get_freesurfer_home



@pytest.fixture
def freesurfer_home():
    """Fixture to set up FreeSurfer home for testing."""
    fs_home = os.environ.get('FREESURFER_HOME')
    if fs_home:
        return Path(fs_home)
    
    # Return the mocked home directory
def test_freesurfer():
    assert get_freesurfer_home().exists(), "FREESURFER_HOME should exist"
    assert get_freesurfer_home().is_dir(), "FREESURFER_HOME should be a directory"

    directory_listing = os.listdir(get_freesurfer_home() / 'subjects')
    assert "bert" in directory_listing, "Expected 'bert' directory in FreeSurfer home"

def test_ScalpelSubject_load():
    from src.classes.subject import ScalpelSubject
    
    ## test loading bert
    subject_directory = Path(get_freesurfer_home()) / "subjects"
    subject = ScalpelSubject(subject_id="bert", subjects_dir = subject_directory, hemi = 'lh')
    assert subject.subject_id == "bert", "Expected subject ID to be 'bert'"
    assert subject.subject_fs_path == get_freesurfer_home() / 'subjects' / 'bert', "Expected FreeSurfer home to be set correctly"
    assert subject.gyrus[0] is not None, "Gyral components not properly identified"

def test_ScalpelSubject_load_label():
    from src.classes.subject import ScalpelSubject

    subject_directory = Path(get_freesurfer_home()) / "subjects"    
    print(subject_directory.resolve())
    subject = ScalpelSubject(subject_id="bert", subjects_dir = subject_directory, hemi = 'lh')
    ## test loading bert labels
    subject.load_label("BA1_exvivo")
    subject.load_label("BA2_exvivo")
    gyrus = subject.labels["BA1_exvivo"].gyrus()
    sulcus = subject.labels["BA2_exvivo"].sulcus()
    assert gyrus is not None, "Gyrus not loading correctly"
    assert sulcus is not None, "Sulcus not loading correctly"

    # test combine label
    subject.combine_labels(["BA1_exvivo", "BA2_exvivo"], "combined_label")
    combined_label = subject.labels["combined_label"]
    assert combined_label is not None, "Combined label not loading correctly"

    # test write label
    subject.write_label(label_name = "combined_label")
    assert os.path.exists(subject.subject_fs_path / "label" / "lh.combined_label.label"), "Label file not created"
    # remove label file
    os.remove(subject.subject_fs_path / "label" / "lh.combined_label.label")

