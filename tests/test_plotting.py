import pytest
import os
from pathlib import Path


@pytest.fixture
def freesurfer_home():
    """Fixture to set up FreeSurfer home for testing."""
    fs_home = os.environ.get('FREESURFER_HOME')
    if fs_home:
        return Path(fs_home)


def test_ScalpelSubject_load(freesurfer_home):
    # Test loading a subject using ScalpelSubject with bert
    from scalpel.subject import ScalpelSubject
    
    subject_directory = Path(freesurfer_home) / "subjects"
    subject = ScalpelSubject(subject_id="bert", subjects_dir = subject_directory, hemi = 'lh')
    assert subject.subject_id == "bert", "Expected subject ID to be 'bert'"

def test_ScalpelSubject_plotting(freesurfer_home):
    # Test plotting a subject using ScalpelSubject with bert
    from scalpel.subject import ScalpelSubject
    
    subject_directory = Path(freesurfer_home) / "subjects"
    subject = ScalpelSubject(subject_id="bert", subjects_dir = subject_directory, hemi = 'lh')
    
    # Test plotting the subject labels

    visualizer = subject.plotter
    scene = visualizer.scene
    
    subject.load_label("BA1_exvivo")
    subject.load_label("BA2_exvivo")
    
    # Test internal plotting methods
    visualizer._plot_label("BA1_exvivo")
    visualizer._plot_label("BA2_exvivo")
    
    # Verify the plotting worked
    assert "BA1_exvivo" in scene.geometry.keys()
    assert "BA2_exvivo" in scene.geometry.keys()
    
    # Optional: render to verify it can be displayed
    try:
        png_data = scene.save_image(resolution=[400, 300])
        print(f"Scene rendered successfully ({len(png_data)} bytes)")
    except:
        print("Headless rendering not available, but scene creation succeeded")