SUBJECTS_DIR='/home/weiner/DevProso/subjects'
SUBJECTS=$(cat $SUBJECTS_DIR/all_subjects_list.txt)

for sub in $SUBJECTS; do
    label_stats_dir=$SUBJECTS_DIR/$sub/label
    mris_anatomical_stats -b -l lh.cortex_gyri.label ${sub} lh > ${label_stats_dir}/lh.cortex_gyri.stats.txt
    mris_anatomical_stats -b -l lh.cortex_sulci.label ${sub} lh > ${label_stats_dir}/lh.cortex_sulci.stats.txt

    mris_anatomical_stats -b -l rh.cortex_gyri.label ${sub} rh > ${label_stats_dir}/rh.cortex_gyri.stats.txt
    mris_anatomical_stats -b -l rh.cortex_sulci.label ${sub} rh > ${label_stats_dir}/rh.cortex_sulci.stats.txt
    

    
    
    
done

