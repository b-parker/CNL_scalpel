import pytest
import os
from pathlib import Path



@pytest.fixture
def freesurfer_home():
    """Fixture to set up FreeSurfer home for testing."""
    fs_home = os.environ.get('FREESURFER_HOME')
    if fs_home:
        return Path(fs_home)
    
    
def test_freesurfer(freesurfer_home):
    #Ensure FreeSurver is properly configured and bert exists
    assert freesurfer_home.exists(), "FREESURFER_HOME should exist"
    assert freesurfer_home.is_dir(), "FREESURFER_HOME should be a directory"

    directory_listing = os.listdir(freesurfer_home / 'subjects')
    assert "bert" in directory_listing, "Expected 'bert' directory in FreeSurfer home"

def test_ScalpelSubject_load(freesurfer_home):
    # Test loading a subject using ScalpelSubject with bert
    from scalpel.subject import ScalpelSubject
    
    subject_directory = Path(freesurfer_home) / "subjects"
    subject = ScalpelSubject(subject_id="bert", subjects_dir = subject_directory, hemi = 'lh')
    assert subject.subject_id == "bert", "Expected subject ID to be 'bert'"
    assert subject.subject_fs_path == freesurfer_home / 'subjects' / 'bert', "Expected FreeSurfer home to be set correctly"
    assert subject.gyrus[0] is not None, "Gyral components not properly identified"

def test_ScalpelSubject_load_label(freesurfer_home):
    # Test loading a label using ScalpelSubject with bert
    from scalpel.subject import ScalpelSubject

    subject_directory = Path(freesurfer_home) / "subjects"    
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

