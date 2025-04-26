import pytest
import os
from pathlib import Path
from scalpel.utilities.fs_config import get_freesurfer_home


@pytest.fixture
def freesurfer_home():
    """Fixture to set up FreeSurfer home for testing."""
    fs_home = os.environ.get('FREESURFER_HOME')
    if fs_home:
        return Path(fs_home)


def test_ScalpelSubject_load():
    # Test loading a subject using ScalpelSubject with bert
    from scalpel.subject import ScalpelSubject
    
    subject_directory = Path(get_freesurfer_home()) / "subjects"
    subject = ScalpelSubject(subject_id="bert", subjects_dir = subject_directory, hemi = 'lh')
    assert subject.subject_id == "bert", "Expected subject ID to be 'bert'"

def test_ScalpelSubject_plotting():
    # Test plotting a subject using ScalpelSubject with bert
    from scalpel.subject import ScalpelSubject
    
    subject_directory = Path(get_freesurfer_home()) / "subjects"
    subject = ScalpelSubject(subject_id="bert", subjects_dir = subject_directory, hemi = 'lh')
    
    # Test plotting the subject labels
    subject.plot()
    subject.load_label("BA1_exvivo")
    subject.load_label("BA2_exvivo")
    
    
    subject.plot(labels=["BA1_exvivo", "BA2_exvivo"])
