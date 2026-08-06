import pytest
import os
import csv
from pathlib import Path


@pytest.fixture
def freesurfer_home():
    """Fixture to set up FreeSurfer home for testing."""
    fs_home = os.environ.get('FREESURFER_HOME')
    if fs_home:
        return Path(fs_home)


def test_run_batch_measurements(freesurfer_home, tmp_path):
    from scalpel.measurement.batch import run_batch_measurements

    subjects_dir = Path(freesurfer_home) / "subjects"
    subject_list_file = tmp_path / "subjects.txt"
    subject_list_file.write_text("bert\ndoes_not_exist_12345\n")
    output_file = tmp_path / "results.csv"

    run_batch_measurements(
        subjects_dir=subjects_dir,
        subject_list_file=subject_list_file,
        labels=["BA1_exvivo", "not_a_real_label"],
        measurements=["depth"],
        output_file=output_file,
        hemis=["lh"],
        n_jobs=1,
    )

    assert output_file.exists()
    with open(output_file) as f:
        rows = {(row["subject_id"], row["label"]): row for row in csv.DictReader(f)}

    assert rows[("bert", "BA1_exvivo")]["status"] == "ok"
    assert float(rows[("bert", "BA1_exvivo")]["sulcal_depth_mm"]) > 0

    assert rows[("bert", "not_a_real_label")]["status"] == "missing_label"
    assert rows[("bert", "not_a_real_label")]["sulcal_depth_mm"] == "nan"

    assert rows[("does_not_exist_12345", "BA1_exvivo")]["status"] == "error"
    assert rows[("does_not_exist_12345", "BA1_exvivo")]["sulcal_depth_mm"] == "ERROR"

    errors_file = tmp_path / "results_errors.csv"
    assert errors_file.exists()
    with open(errors_file) as f:
        error_rows = list(csv.DictReader(f))
    assert any(r["measurement"] == "subject_load" for r in error_rows)


def test_run_batch_measurements_per_label(freesurfer_home, tmp_path):
    from scalpel.measurement.batch import run_batch_measurements

    subjects_dir = Path(freesurfer_home) / "subjects"
    subject_list_file = tmp_path / "subjects.txt"
    subject_list_file.write_text("bert\n")
    output_file = tmp_path / "results.csv"

    run_batch_measurements(
        subjects_dir=subjects_dir,
        subject_list_file=subject_list_file,
        labels={"BA1_exvivo": ["depth", "area"], "BA2_exvivo": ["depth"]},
        output_file=output_file,
        hemis=["lh"],
        n_jobs=1,
    )

    with open(output_file) as f:
        rows = {row["label"]: row for row in csv.DictReader(f)}

    # BA1_exvivo asked for depth + area -> both populated
    assert rows["BA1_exvivo"]["sulcal_depth_mm"] not in ("", None)
    assert rows["BA1_exvivo"]["surface_area_mm2"] not in ("", None)

    # BA2_exvivo only asked for depth -> area column present (shared CSV
    # schema) but blank on this row, not NaN and not ERROR
    assert rows["BA2_exvivo"]["sulcal_depth_mm"] not in ("", None)
    assert rows["BA2_exvivo"]["surface_area_mm2"] == ""

    # passing both a dict of labels and a flat measurements list is ambiguous
    with pytest.raises(ValueError):
        run_batch_measurements(
            subjects_dir=subjects_dir,
            subject_list_file=subject_list_file,
            labels={"BA1_exvivo": ["depth"]},
            measurements=["area"],
            output_file=output_file,
            n_jobs=1,
        )
