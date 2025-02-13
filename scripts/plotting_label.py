from pathlib import Path
from src.classes.subject import ScalpelSubject


mount_point = "/Users/benparker/Desktop/cnl/neurocluster"
subject_dir = Path(mount_point) / "weiner" / "HCP" / "subjects"
subject = "fsaverage"

fsav_rh = ScalpelSubject(subject, 'rh', subjects_dir=subject_dir)

custom_label_path = Path(mount_point) / "weiner" / "HCP" / "projects" / "cortical_viz" / "prob_maps" / "ALL_SUBS" / "rh.MPM_all_subjects_incl_PROB_pmfs_p.label"
label_name = "MPM_all_subjects_incl_PROB_pmfs_p"
fsav_rh.load_label(label_name=label_name, custom_label_path=custom_label_path)

fsav_rh.threshold_label(label_name, threshold = .25, load_label= True, new_name = f"test_thresh_25_pmfs_p")

fsav_rh.plot_label(f"test_thresh_25_pmfs_p", face_colors='green')