"""

this code returns the depth of a sulcus in mm given an annotation file, subject directory,
and a list of subjects. adapted for python from
Madan, C. R. (2019). Robust estimation of sulcal morphology. Brain Informatics 6, 5.
            doi:10.1186/s40708-019-0098-1

"""

import math
from dataclasses import dataclass, field

import pathlib

import numpy as np
import nibabel as nb
import pandas as pd


@dataclass
class SubjectHemi:
    """
    dataclass that stores information about the current hemi that we are on
    """
    inflated_v: np.array = field(default_factory=lambda: np.array([]))
    inflated_f: np.array = field(default_factory=lambda: np.array([]))
    label_v: str = ""
    annot_name: str = ""
    current_hemi: str = ""
    pial_v: str = ""
    sulc_map: str = ""
    gyrif_f: str = ""
    gyrif_v: str = ""
    faces: str = ""


@dataclass()
class Outputs:
    """
    the outputs that we want, the list of sulci & depths
    used by the save function
    """

    fname: str = ""
    list_subjects: np.array = None
    sulc_list: np.array = None
    sulc_depth: np.array = None
    hemis: np.array = None
    subjects: np.array = None


#  defining functions
def calc_sulc(subjects, subjects_dir, hemis, sulc_list, output, fundus=True):
    """
    calculates the depth of the sulci of interest, and calls save() at the end

    parameters
    --------------
    subjects: list of subjects to check
    subjects_dir: path to the directory of subjects
    hemis: which hemis we are checking ex: [lh, rh]
    sulc_list: list of sulci we are calculating depth for
    output: output data class that stores output data
    fundus: boolean, true for calculating based on fundus, false if using entire sulcus

    returns
    -------------
    output data class, now mutated with new data
    """

    # if the user wants all subjects in a directory, they can input "."
    if subjects[0] == ".":
        list_subjects_paths = get_subject_list(subjects_dir)
        list_subjects = []
        for sub in list_subjects_paths:
            list_subjects.append(sub.name)
    else:
        list_subjects = subjects

    output.list_subjects = list_subjects
    output.hemis = []
    output.sulc_list = []
    output.sulc_depth = []
    output.subjects = []

    for sub in list_subjects:
        print("calculating sulci for subject " + sub)
        count = 0

        # loops through the hemis
        for hemi in hemis:
            # hemi = str(hemi) - in matlab it's casted, might not need to.
            print(hemi)
            subject_hemi = load(subjects_dir, sub, hemi)

            for sulc in sulc_list:
                # isolate mesh for specific sulcus
                count += 1
                print(count, ":", sulc)
                mesh = isolate(subjects_dir, subject_hemi, sulc, sub, hemi)
                if mesh is np.nan:
                    sulci_d = np.nan
                else:
                    label_v = mesh[0]
                    sulci_d = depth(subject_hemi, label_v, fundus)
                # continue to next sulci
                output.subjects.append(sub)
                output.hemis.append(hemi)
                output.sulc_list.append(sulc)
                output.sulc_depth.append(sulci_d)
    save(output)
    print("done")
    return output


def get_subject_list(subjects_dir):
    """
    makes a list of the subjects in a directory

    parameters
    --------------
    subjects_dir : a path to the subject directory

    returns
    --------------
    list of all subjects as an array

    """

    # subjects_dir is a directory, subjects is an array.
    subjects_dir = pathlib.Path(subjects_dir)
    list_subjects = []
    for item in (subjects_dir.iterdir()):
        if item.is_dir():
            list_subjects.append(item)
    return list_subjects


def name_file(file_string, file_type):
    """
    names a file

    parameters
    --------------
    file_string: the first part of the file name, usually the hemisphere.
    file_type: the type of file, ex. "pial" or "inflated"

    returns
    -------
    a string with the name of the file. ex. lh.pial

    """
    return file_string + "." + file_type


def read_label(label_name):
    """
    Reads a freesurfer-style .label file (5 columns)

    Parameters
    ----------
    label_name: str

    Returns
    -------
    vertices: index of the vertex in the label np.array [n_vertices]
    RAS_coords: columns are the X,Y,Z RAS coords associated with vertex number in the label,
                np.array [n_vertices, 3]

    """

    # read label file, excluding first two lines of descriptor
    df_label = pd.read_csv(label_name, skiprows=[0, 1], header=None,
                           names=['vertex', 'x_ras', 'y_ras', 'z_ras', 'stat'], delimiter=r'\s+')

    vertices = np.array(df_label.vertex)
    RAS_coords = np.empty(shape=(vertices.shape[0], 3))
    RAS_coords[:, 0] = df_label.x_ras
    RAS_coords[:, 1] = df_label.y_ras
    RAS_coords[:, 2] = df_label.z_ras

    return vertices, RAS_coords


def load(subject_dir, subject, hemi):
    """
    reading the labels into python, using the toolbox given by FreeSurfer

    parameters
    -----------
    subject_dir: path to the directory of subjects.
    subject: the subject to load
    hemi: which hemisphere to load

    returns
    -------
    subject_hemi, a SubjectHemi dataclass that contains the information for the hemisphere we are calculating
        depth for.

    """

    # reading the labels into python, using the toolbox given by FreeSurfer
    # turning those file names into paths
    def makepath(file_type):
        file_name = pathlib.Path(subject_dir, subject, 'surf', file_type)
        return file_name

    file_pial = makepath(name_file(hemi, "pial"))
    file_inflated = makepath(name_file(hemi, "inflated"))
    file_pial_smoothed = makepath(name_file(hemi, "pial-outer-smoothed"))
    file_sulc = makepath(name_file(hemi, "sulc"))

    # read in the info using nibabel

    subject_hemi = SubjectHemi()

    pial_verts, pial_faces = nb.freesurfer.read_geometry(str(file_pial))
    inflated_verts, inflated_faces = nb.freesurfer.read_geometry(str(file_inflated))
    sulc_map = nb.freesurfer.read_morph_data(str(file_sulc))
    gyrif_verts, gyrif_faces = nb.freesurfer.read_geometry(str(file_pial_smoothed))

    subject_hemi.pial_v = pial_verts
    subject_hemi.faces = pial_faces  # f gets reassigned here
    subject_hemi.inflated_v = inflated_verts
    subject_hemi.inflated_f = inflated_faces
    subject_hemi.sulc_map = sulc_map
    subject_hemi.gyrif_v = gyrif_verts
    subject_hemi.gyrif_f = gyrif_faces

    return subject_hemi


def get_faces_from_vertices(faces: np.array, label_ind: np.array):
    """
    Takes a list of faces and label indices
    Returns the faces that contain the indices

    parameters
    __________
    faces: array of faces composed of 3 points
    label_ind: array of indices of points in the label
        (first column of label file; 0 index in read_label)

    output
    _________
    label_faces: array of faces that contain the points in the label
    """

    all_label_faces = []
    for face in faces:
        for point_index in face:
            if point_index in label_ind:
                all_label_faces.append(list(face))
    return np.array(all_label_faces)


def isolate(subject_dir: str, subject_hemi: object, label_name: str, subject: str, hemi: str):
    """
    Isolating a sulcus mesh from an entire subject hemisphere

    parameters
    ----------
    subject_hemi : object - SubjectHemi object
    label_name : str - the name of the label file i.e. 'POS' for rh.POS.label
    subject_dir: path to the directory of subjects
    hemi: which hemi we're on
    subject: which subject we're on

    returns
    -------
    Returns the mesh for the sulcus of label_name within subject_hemi

    """

    # isolating the faces associated with each individual sulcus as a 3D mesh.
    # id path for sulcus file, assert it exists
    fname = pathlib.Path(subject_dir, subject, 'label', f"{hemi}.{label_name}.label")
    if fname.exists():
        label_v = read_label(str(fname))[0]
        isolated_faces = get_faces_from_vertices(subject_hemi.faces, label_v)
    else:
        print(f'no sulcus at {fname}')
        return np.NaN
    return label_v, isolated_faces


def depth(subject_hemi, label_v, fundus):
    """
    calculates Euclidean distance from fundus to smoothed surface generated by FreeSurfer
    sulcal fundus - 100 vertices with the lowest values on the sulcal map. (if fundus is false, it calculates
    for all vertices)

    parameters
    ------------
    fundus: boolean, do you want to average the depth of the entire sulcus or just the fundus?
    label_v: the label we are finding the depth of
    subject_hemi: subject_hemi dataclass of the current hemi we are on

    returns
    ---------
    depth for a sulcus (the sulcus stored as label_v in subject_hemi)

    """

    sorted_label_sulcval = np.argsort(subject_hemi.sulc_map[label_v], kind="mergesort")

    if fundus:
        sorted_label_sulcval = sorted_label_sulcval[-100:]

    depth_array = []
    for index in sorted_label_sulcval:
        v_xyz = subject_hemi.pial_v[index]
        min_depth = None
        for point in subject_hemi.gyrif_v:
            new_distance = math.dist(v_xyz, point)
            if min_depth is None or min_depth > new_distance:
                min_depth = new_distance
        depth_array.append(min_depth)
    depth_final = np.median(depth_array)
    return depth_final


def save(save_output):
    """
    param save_output: output dataclass that includes all the info that will be in the csv
    changes the information in save_output into a csv format.
    """


    save_data = {'subject': save_output.subjects, 'sulcus': save_output.sulc_list, 'hemi': save_output.hemis, 'depth': save_output.sulc_depth}
    tbl = pd.DataFrame(save_data)


    tbl.to_csv(save_output.fname + '_sulcidepth.csv', index=False)


def main():
    """
    enter your annot name, subject_dir, and sulc_list, subject_list, and which hemi
        you want here.
    outputs a csv file with the depths for the sulci, hemis, and subjects inputted.
    outputs np.NaN if sulcus is not present.
    """
    from pathlib import Path
    from src.utilities.freesurfer_utils import get_subjects_list
    local_subjects_dir = "/Users/benparker/Desktop/cnl/neurocluster/weiner/HCP/subjects"
    subject_list = get_subjects_list("/Users/benparker/Desktop/cnl/neurocluster/weiner/HCP/subject_lists/HCP_processed_subs_all.txt", local_subjects_dir)[:2]
    print(subject_list)
    # user: sets info here
    fundus = True
    subject_dir = "/home/weiner/Urgency/subjects"
    sulc_list = ['MCGS', 'POS']  # what sulci do you want to check?
    subject_list = [Path(i).name for i in subject_list]
    hemis = ["lh", "rh"]  # which hemis are you checking?
    output = Outputs(fname="HCP_test")  # what do you want the output name to be?
    calc_sulc(subject_list, local_subjects_dir, hemis, sulc_list, output, fundus)


if __name__ == "__main__":
    main()

