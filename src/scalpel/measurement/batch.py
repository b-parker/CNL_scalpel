"""
Batch morphometry: run CNL_scalpel measurements across a subject list and
multiple sulcal labels, in parallel, writing results to a CSV.
"""
from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def _measurement_columns(measurement: str) -> List[str]:
    return {
        'area': ['surface_area_mm2'],
        'thickness': ['thickness_mean_mm', 'thickness_std_mm'],
        'depth': ['sulcal_depth_mm'],
        'length': ['sulcal_length_mm'],
        'width': ['sulcal_width_mm'],
        'volume': ['gray_volume_mm3'],
        'curvature': ['mean_curvature', 'gaussian_curvature'],
        'indices': ['folding_index', 'intrinsic_curvature_index'],
        'all_freesurfer': ['num_vertices', 'surface_area_mm2', 'gray_volume_mm3',
                            'thickness_mean_mm', 'thickness_std_mm', 'mean_curvature',
                            'gaussian_curvature', 'folding_index', 'intrinsic_curvature_index'],
    }[measurement]


def _compute_measurement(subject, label_name: str, measurement: str) -> Dict[str, float]:
    """Compute one measurement's column(s) for an already-loaded label. Raises on failure --
    callers are responsible for catching and flagging errors per-measurement."""
    if measurement == 'area':
        return {'surface_area_mm2': subject.calculate_surface_area(label_name)}
    elif measurement == 'thickness':
        mean_t, std_t = subject.calculate_cortical_thickness(label_name)
        return {'thickness_mean_mm': mean_t, 'thickness_std_mm': std_t}
    elif measurement == 'depth':
        return {'sulcal_depth_mm': subject.calculate_sulcal_depth(label_name)}
    elif measurement == 'length':
        return {'sulcal_length_mm': subject.calculate_sulcal_length(label_name)}
    elif measurement == 'width':
        return {'sulcal_width_mm': subject.calculate_sulcal_width(label_name)}
    elif measurement == 'volume':
        return {'gray_volume_mm3': subject.calculate_gray_matter_volume(label_name)}
    elif measurement == 'curvature':
        return {
            'mean_curvature': subject.calculate_absolute_curvature(label_name, 'mean'),
            'gaussian_curvature': subject.calculate_absolute_curvature(label_name, 'gaussian'),
        }
    elif measurement == 'indices':
        fold_idx, intrinsic_idx = subject.calculate_curvature_indices(label_name)
        return {'folding_index': fold_idx, 'intrinsic_curvature_index': intrinsic_idx}
    elif measurement == 'all_freesurfer':
        stats = subject.calculate_all_freesurfer_stats(label_name)
        return {
            'num_vertices': stats['num_vertices'], 'surface_area_mm2': stats['surface_area_mm2'],
            'gray_volume_mm3': stats['gray_volume_mm3'], 'thickness_mean_mm': stats['thickness_mean_mm'],
            'thickness_std_mm': stats['thickness_std_mm'], 'mean_curvature': stats['mean_curvature'],
            'gaussian_curvature': stats['gaussian_curvature'], 'folding_index': stats['folding_index'],
            'intrinsic_curvature_index': stats['intrinsic_curvature_index'],
        }
    else:
        raise ValueError(f"Unknown measurement '{measurement}'")


def _label_path(subjects_dir: str, subject_id: str, hemi: str, label_name: str) -> Path:
    return Path(subjects_dir) / subject_id / 'label' / f'{hemi}.{label_name}.label'


def _process_subject_hemi(subject_id: str, hemi: str, subjects_dir: str, surface_type: str,
                          label_measurements: Dict[str, List[str]]) -> Tuple[List[dict], List[dict]]:
    """
    Runs (possibly in a worker process): compute each label's own requested
    measurements for one subject/hemisphere. Returns (rows, errors) -- rows is
    one dict per label ready to write to the results CSV, errors is one dict
    per (label, measurement) failure ready to write to the error log.

    A row only contains columns for the measurements requested for that
    particular label; a measurement not requested for a given label is simply
    absent from its row dict and shows up blank in the CSV, distinct from
    NaN ('measurement was requested but the label doesn't exist') and 'ERROR'
    ('measurement was requested and failed').
    """
    from scalpel.subject import ScalpelSubject  # imported inside: must be picklable/importable in worker processes

    rows: List[dict] = []
    errors: List[dict] = []

    try:
        subject = ScalpelSubject(subject_id=subject_id, hemi=hemi, subjects_dir=subjects_dir,
                                 surface_type=surface_type)
    except Exception as exc:
        for label_name, measurements in label_measurements.items():
            columns = [col for m in measurements for col in _measurement_columns(m)]
            rows.append({'subject_id': subject_id, 'hemi': hemi, 'label': label_name,
                         'status': 'error', **{c: 'ERROR' for c in columns}})
            errors.append({'subject_id': subject_id, 'hemi': hemi, 'label': label_name,
                           'measurement': 'subject_load', 'error': f'{type(exc).__name__}: {exc}'})
        return rows, errors

    for label_name, measurements in label_measurements.items():
        columns = [col for m in measurements for col in _measurement_columns(m)]

        if not _label_path(subjects_dir, subject_id, hemi, label_name).exists():
            rows.append({'subject_id': subject_id, 'hemi': hemi, 'label': label_name,
                         'status': 'missing_label', **{c: float('nan') for c in columns}})
            continue

        try:
            subject.load_label(label_name)
        except Exception as exc:
            rows.append({'subject_id': subject_id, 'hemi': hemi, 'label': label_name,
                         'status': 'error', **{c: 'ERROR' for c in columns}})
            errors.append({'subject_id': subject_id, 'hemi': hemi, 'label': label_name,
                           'measurement': 'load_label', 'error': f'{type(exc).__name__}: {exc}'})
            continue

        row = {'subject_id': subject_id, 'hemi': hemi, 'label': label_name}
        any_error = False
        for measurement in measurements:
            try:
                row.update(_compute_measurement(subject, label_name, measurement))
            except Exception as exc:
                any_error = True
                for col in _measurement_columns(measurement):
                    row[col] = 'ERROR'
                errors.append({'subject_id': subject_id, 'hemi': hemi, 'label': label_name,
                               'measurement': measurement, 'error': f'{type(exc).__name__}: {exc}'})
        row['status'] = 'error' if any_error else 'ok'
        rows.append(row)

    return rows, errors


def _read_subject_ids(subject_list_file: Union[str, Path]) -> List[str]:
    with open(subject_list_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def _normalize_label_measurements(labels: Union[List[str], Dict[str, List[str]]],
                                  measurements: Optional[List[str]]) -> Dict[str, List[str]]:
    if isinstance(labels, dict):
        if measurements is not None:
            raise ValueError("Pass either `labels` as a dict mapping each label to its own measurement "
                             "list, or `labels` as a flat list with `measurements` applied to all of "
                             "them -- not both.")
        return labels
    if measurements is None:
        raise ValueError("`measurements` is required when `labels` is a flat list of label names; "
                         "pass `labels` as a dict of {label_name: [measurements]} instead for "
                         "per-label measurement selection.")
    return {label_name: list(measurements) for label_name in labels}


def run_batch_measurements(
    subjects_dir: Union[str, Path],
    subject_list_file: Union[str, Path],
    labels: Union[List[str], Dict[str, List[str]]],
    output_file: Union[str, Path],
    measurements: Optional[List[str]] = None,
    hemis: List[str] = ['lh', 'rh'],
    surface_type: str = 'white',
    n_jobs: int = 1,
) -> None:
    """
    Run CNL_scalpel measurements across a subject list and multiple sulcal
    labels, writing a long-format CSV (one row per subject x hemi x label).

    Parameters:
    -----------
    subjects_dir: Union[str, Path]
        FreeSurfer SUBJECTS_DIR containing each subject's recon-all output.
    subject_list_file: Union[str, Path]
        Path to a .txt file with one subject ID per line (matching the
        subdirectory names under subjects_dir).
    labels: Union[List[str], Dict[str, List[str]]]
        Either a flat list of sulcal label names (without the hemi prefix),
        e.g. ['PCGS', 'mfs', 'POS'], in which case `measurements` is required
        and applies to every label the same way; or a dict mapping each label
        to its own measurement list, e.g. {'PCGS': ['length', 'width',
        'depth'], 'mfs': ['depth', 'area']}, for when a measurement isn't
        meaningful for every sulcus (length is slow and not always useful,
        for instance). `measurements` must be left as None in this case.
    output_file: Union[str, Path]
        Path to the output CSV. A companion file with '_errors' inserted
        before the extension is also written whenever any measurement fails,
        with the actual exception message per (subject, hemi, label,
        measurement) -- the main CSV only shows 'ERROR' in the failed cell(s).
    measurements: Optional[List[str]]
        Measurements to compute for every label in `labels`, when `labels` is
        a flat list: 'area', 'thickness', 'depth', 'length', 'width',
        'volume', 'curvature', 'indices', 'all_freesurfer' (same vocabulary as
        ScalpelMeasurer.export_measurements). Leave as None when `labels` is
        already a dict of per-label measurement lists.
    hemis: List[str]
        Hemispheres to process (default both).
    surface_type: str
        Surface used to construct each ScalpelSubject. Doesn't affect the
        values above -- sulcal depth/width use the pial surface, length uses
        the fiducial surface, area/volume/thickness use white/pial directly,
        all independent of surface_type -- but a surface file for this type
        must still exist to load the subject at all.
    n_jobs: int
        Number of worker processes. 1 (default) runs serially in the calling
        process, which is easiest to debug. -1 uses all available CPU cores.
        Each task is one (subject, hemi) pair -- loading a subject and
        building its geodesic solver is itself expensive, so a worker reuses
        both across every requested label for that hemisphere rather than
        reloading per label.

    NOTE: on macOS/Windows, multiprocessing uses the 'spawn' start method, so
    if calling this with n_jobs != 1 from a plain script (not a notebook),
    the call must be inside an ``if __name__ == "__main__":`` guard or the
    worker processes will try to re-import and re-run the calling script.

    Rows are written incrementally as each (subject, hemi) task completes, so
    an interrupted run still leaves completed results on disk.

    Row 'status' is one of:
    - 'ok': every requested measurement computed successfully.
    - 'missing_label': the label file doesn't exist for this subject/hemi
      (checked directly against the label directory, not inferred from a
      crash); every measurement column requested for that label is NaN.
    - 'error': the label exists but at least one measurement failed; failed
      measurement cells contain the string 'ERROR' (successful measurements
      in the same row still show their real value), and the failure is
      logged with its actual exception message in the errors CSV.

    A measurement column is left blank (not NaN, not 'ERROR') on rows for
    labels that didn't request it -- e.g. if only 'PCGS' asked for 'length',
    the sulcal_length_mm cell is blank on every other label's rows.
    """
    label_measurements = _normalize_label_measurements(labels, measurements)

    subject_ids = _read_subject_ids(subject_list_file)
    tasks = [(subject_id, hemi) for subject_id in subject_ids for hemi in hemis]

    output_file = Path(output_file)
    errors_file = output_file.with_name(f"{output_file.stem}_errors{output_file.suffix}")

    columns: List[str] = []
    for m_list in label_measurements.values():
        for m in m_list:
            for col in _measurement_columns(m):
                if col not in columns:
                    columns.append(col)
    fieldnames = ['subject_id', 'hemi', 'label', 'status'] + columns
    error_fieldnames = ['subject_id', 'hemi', 'label', 'measurement', 'error']

    n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    print(f"Running {len(tasks)} (subject, hemi) tasks "
          f"{'serially' if n_workers == 1 else f'across {n_workers} workers'}...")

    with open(output_file, 'w', newline='') as out_f, open(errors_file, 'w', newline='') as err_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        error_writer = csv.DictWriter(err_f, fieldnames=error_fieldnames)
        error_writer.writeheader()

        n_done = 0
        t_start = time.time()

        def handle_result(subject_id, hemi, rows, errors):
            nonlocal n_done
            n_done += 1
            writer.writerows(rows)
            out_f.flush()
            if errors:
                error_writer.writerows(errors)
                err_f.flush()
            elapsed = time.time() - t_start
            print(f"[{n_done}/{len(tasks)}] {subject_id} {hemi} done "
                  f"({len(errors)} errors) -- {elapsed:.0f}s elapsed", flush=True)

        if n_workers == 1:
            for subject_id, hemi in tasks:
                rows, errors = _process_subject_hemi(subject_id, hemi, str(subjects_dir), surface_type,
                                                      label_measurements)
                handle_result(subject_id, hemi, rows, errors)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_process_subject_hemi, subject_id, hemi, str(subjects_dir), surface_type,
                                    label_measurements): (subject_id, hemi)
                    for subject_id, hemi in tasks
                }
                for future in as_completed(futures):
                    subject_id, hemi = futures[future]
                    try:
                        rows, errors = future.result()
                    except Exception as exc:
                        rows = [{'subject_id': subject_id, 'hemi': hemi, 'label': label_name,
                                 'status': 'error',
                                 **{c: 'ERROR' for m in m_list for c in _measurement_columns(m)}}
                                for label_name, m_list in label_measurements.items()]
                        errors = [{'subject_id': subject_id, 'hemi': hemi, 'label': '',
                                   'measurement': 'worker_process', 'error': f'{type(exc).__name__}: {exc}'}]
                    handle_result(subject_id, hemi, rows, errors)

    print(f"Wrote results to {output_file}")
    if errors_file.exists() and errors_file.stat().st_size > len(','.join(error_fieldnames)) + 2:
        print(f"Wrote errors to {errors_file}")
