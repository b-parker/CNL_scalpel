
#! /usr/bin/bash

SUBJECTS_DIR=/home/weiner/HCP/subjects
cd $SUBJECTS_DIR

sub=fsaverage   #${SGE_TASK}
echo ${sub}

#make dir to save labels in, if dne 
SAVE_LABELS_DIR=$SUBJECTS_DIR/${sub}/label/MPM_labels/

LABEL_DIR=/home/weiner/HCP/projects/cortical_viz/prob_maps/ALL_SUBS

# make dir for mni labels

SAVE_MNI_VOLS_DIR=$SUBJECTS_DIR/${sub}/label/cortical_viz_MPM_vols_T1_filled/FLIRT_MNI/
if [[ ! -e $SAVE_MNI_VOLS_DIR ]]; then
    mkdir -p $SAVE_MNI_VOLS_DIR; echo 'making dir: ' $SAVE_LABELS_DIR; fi

# Set paths to required files
MNI_BRAIN=$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz
SUBJECT_DIR=$SUBJECTS_DIR 
OUTPUT_DIR=$SAVE_MNI_VOLS_DIR
MASKS=($(cat /home/weiner/HCP/projects/cortical_viz/prob_maps/ALL_SUBS/all_projection_names.txt | tr -d '\r' | sed 's/[[:space:]]*$//' ))
SUBJECT_T1_BRAIN="${SUBJECT_DIR}/$sub/mri/T1.mgz"
VOLS_DIR=/home/weiner/HCP/subjects/fsaverage/label/cortical_viz_MPM_vols_T1_filled/final

HEMIS=("lh" "rh")

# Convert subject's T1 brain image from mgz to nifti 
mri_convert "$SUBJECT_T1_BRAIN" "${SUBJECT_DIR}/$sub/mri/T1.nii.gz"

# Compute xform mat using flirt
flirt -in "${SUBJECT_DIR}/$sub/mri/T1.nii.gz" \
      -ref "$MNI_BRAIN" \
      -omat "${OUTPUT_DIR}/native2mni_1mm.mat" \
      -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear

# Apply xform to each mask
for MASK in "${MASKS[@]}"; do
    for hemi in "${HEMIS[@]}"; do
	echo  "MASK FILE: $VOLS_DIR/${hemi}.${MASK}_centroid_final.nii.gz" 

  flirt -in "$VOLS_DIR/${hemi}.${MASK}_centroid_final.nii.gz" \
        -ref "$MNI_BRAIN" \
        -applyxfm -init "${OUTPUT_DIR}/native2mni_1mm.mat" \
        -out "${OUTPUT_DIR}/${hemi}.${MASK}_MNI_1mm.nii" \
        -interp nearestneighbour
    done
    done

echo "Registration complete! The registered masks are saved in $OUTPUT_DIR."


