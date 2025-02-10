#! /usr/bin/bash

SUBJECTS_DIR=/home/weiner/HCP/subjects
cd $SUBJECTS_DIR

sub=fsaverage   #${SGE_TASK}
echo ${sub}

#make dir to save labels in, if dne 
SAVE_LABELS_DIR=$SUBJECTS_DIR/${sub}/label/MPM_labels/
if [[ ! -e $SAVE_LABELS_DIR ]]; then
    mkdir $SAVE_LABELS_DIR; echo 'making dir: ' $SAVE_LABELS_DIR; fi

#labels to be converted
array1=$(cat /home/weiner/HCP/projects/cortical_viz/prob_maps/ALL_SUBS/all_projection_names.txt)

echo "Loaded labels."

LABEL_DIR=/home/weiner/HCP/projects/cortical_viz/prob_maps/ALL_SUBS

#make dir to save vols in (intermediate & final steps), if dne 
SAVE_VOLS_DIR=$SUBJECTS_DIR/${sub}/label/cortical_viz_MPM_vols_T1_filled/
if [[ ! -e $SAVE_VOLS_DIR ]]; then
    mkdir -p  $SAVE_VOLS_DIR; echo 'making dir: ' $SAVE_LABELS_DIR; fi
FINAL_VOLS_DIR=$SUBJECTS_DIR/${sub}/label/cortical_viz_MPM_vols_T1_filled/final/
if [[ ! -e $FINAL_VOLS_DIR ]]; then
    mkdir -p  $FINAL_VOLS_DIR; echo 'making dir: ' $SAVE_LABELS_DIR; fi


# 1) Convert labels to a vol in the same space as the labels first
#for label_name in ${labels[@]}; do
for hemi in 'lh' 'rh'; do
  echo $hemi
  for label in ${array1[@]}; do
    
    label_name=$label
    echo $label_subdir
    echo $label_name
    echo "Beginning mri_label2vol"

    echo "Label :  $LABEL_DIR/${label_subdir}/${hemi}.${label_name}.label"
    
    mri_label2vol \
      --label $LABEL_DIR/${label_subdir}/${hemi}.${label_name}.label \
      --subject ${sub} \
      --hemi ${hemi} \
      --temp $SUBJECTS_DIR/$sub/mri/T1.mgz \
      --identity \
      --proj frac 0 1 0.01 \
      --o $SAVE_VOLS_DIR/${hemi}.${label_name}_proj.nii.gz

    # fill in ribbon
    mri_binarize --dilate 1 --erode 1 --i $SAVE_VOLS_DIR/${hemi}.${label_name}_proj.nii.gz --o $SAVE_VOLS_DIR/${hemi}.${label_name}_filled.nii.gz --min 1
    mris_calc -o $FINAL_VOLS_DIR/${hemi}.${label_name}_final.nii.gz $SAVE_VOLS_DIR/${hemi}.${label_name}_filled.nii.gz mul $SUBJECTS_DIR/$sub/mri/$hemi.ribbon.mgz

  done
done



# 2) move to mni space

# make dir for mni labels
SAVE_MNI_VOLS_DIR=$SUBJECTS_DIR/${sub}/label/label_vols_T1_filled/FLIRT_MNI/
if [[ ! -e $SAVE_MNI_VOLS_DIR ]]; then
    mkdir $SAVE_MNI_VOLS_DIR; echo 'making dir: ' $SAVE_LABELS_DIR; fi

# Set paths to required files
MNI_BRAIN=$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz
SUBJECT_DIR=$SUBJECTS_DIR 
OUTPUT_DIR=$SAVE_MNI_VOLS_DIR
MASKS=("lh.HCP_PMC_PROB_MPM_binary_0.1_MCGS" "rh.HCP_PMC_PROB_MPM_binary_0.1_MCGS")  # List of volumes to be registered to mni
SUBJECT_T1_BRAIN="${SUBJECT_DIR}/$sub/mri/T1.mgz"

# Convert subject's T1 brain image from mgz to nifti 
mri_convert "$SUBJECT_T1_BRAIN" "${SUBJECT_DIR}/$sub/mri/T1.nii.gz"

# Compute xform mat using flirt
flirt -in "${SUBJECT_DIR}/$sub/mri/T1.nii.gz" \
      -ref "$MNI_BRAIN" \
      -omat "${OUTPUT_DIR}/native2mni_1mm.mat" \
      -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear

# Apply xform to each mask
for MASK in "${MASKS[@]}"; do
  flirt -in "$VOLS_DIR/${MASK}.nii" \
        -ref "$MNI_BRAIN" \
        -applyxfm -init "${OUTPUT_DIR}/native2mni_1mm.mat" \
        -out "${OUTPUT_DIR}/${MASK}_MNI_1mm.nii" \
        -interp nearestneighbour
done

echo "Registration complete! The registered masks are saved in $OUTPUT_DIR."


