import pytest
import os
from pathlib import Path


@pytest.fixture
def freesurfer_home():
    """Fixture to set up FreeSurfer home for testing."""
    fs_home = os.environ.get('FREESURFER_HOME')
    if fs_home:
        return Path(fs_home)


def test_calculate_sulcal_length(freesurfer_home):
    # Test sulcal length on a real label using ScalpelSubject with bert
    from scalpel.subject import ScalpelSubject

    subject_directory = Path(freesurfer_home) / "subjects"
    subject = ScalpelSubject(subject_id="bert", subjects_dir=subject_directory, hemi='lh')

    subject.load_label("BA1_exvivo")
    length = subject.calculate_sulcal_length("BA1_exvivo")

    assert length > 0, "Sulcal length should be a positive distance in mm"
    assert subject.labels["BA1_exvivo"].measurements['sulcal length (mm)'] == length


def test_fiducial_surface_is_midpoint(freesurfer_home):
    import numpy as np
    from scalpel.subject import ScalpelSubject

    subject_directory = Path(freesurfer_home) / "subjects"
    subject = ScalpelSubject(subject_id="bert", subjects_dir=subject_directory, hemi='lh')

    expected = (subject.white_v + subject.pial_v) / 2.0
    assert np.allclose(subject.fiducial_v, expected)
